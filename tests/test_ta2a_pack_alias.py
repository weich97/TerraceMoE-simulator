# -*- coding: utf-8 -*-
"""并包解包不得与缓冲共享存储(2026-08-21 判决床 rank61 崩溃的回归测试)。

事故经过:
  判决床 pack-on 臂两发都在 iter 30 附近死掉,报
      RuntimeError: The tensor has a non-zero number of elements,
                    but its data is not allocated yet.
      at terrace/ta2a_dispatch.py `owner = slot_idx // epr`
  另外 15 台在集合通信上等这个已经死掉的 rank,表现成"整框挂死 40 分钟"。

根因:
  `hopa_unpack` 用 `.contiguous()` 连续化列切片。PyTorch 判定连续性时,**dim0 尺寸
  为 1 的那一维 stride 被忽略** —— 于是 `buf[:, a:b]` 在 R==1 时被判为已连续,
  `.contiguous()` 原样交回视图。调用方紧接着 `_rbuf.untyped_storage().resize_(0)`
  还内存,三个平面当场悬空,直到十几个算子之后第一次真读才炸。

  为什么对齐床(4 节点)没撞上、判决床(16 节点)撞上了:行被切成 16 份而不是 4 份,
  加上 load_cv 随训练从 1.0 爬到 1.4,某个 rank 收到恰好 1 行的概率从可忽略变成会发生。
  为什么 iter 30 而不是 iter 1:路由要先专门化,分布才够偏。

修法:`ta2a_pack._own()` 用 `clone(memory_format=contiguous_format)`,**永远**新存储;
      调用点还内存前再加一道 `assert_not_aliased` 绊线。

本测试是纯 CPU 的形状级复现,不需要 NPU,也不需要分布式。
"""
import pytest
import torch

from terrace import ta2a_pack as pk


# 判决床几何:H=2048, quota=3(k=6 / M=2), slots=24(384 专家 / 128 EP / 8 rpn)
JUDGMENT = dict(hidden=2048, quota=3, slots=24)

# R=1 是危险形状(dim0 尺寸 1 -> stride 被忽略 -> 切片被判连续);
# R=0 与 R>=2 也一并钉住,免得将来有人"只修 1"。
ROW_COUNTS = [0, 1, 2, 3, 17]


def _storage_ptr(t):
    return t.untyped_storage().data_ptr()


def _aliases(plane, buf):
    """plane 是否与 buf 共享存储。空张量不算(它没有可悬空的数据)。"""
    return plane.numel() > 0 and _storage_ptr(plane) == _storage_ptr(buf)


@pytest.mark.parametrize("rows", ROW_COUNTS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_hopa_unpack_never_aliases_buffer(rows, dtype):
    H, q = JUDGMENT["hidden"], JUDGMENT["quota"]
    payload = torch.zeros(rows, H, dtype=dtype)
    gate = torch.zeros(rows, q, dtype=dtype)
    ids = torch.zeros(rows, q, dtype=torch.int64)

    buf, lay = pk.hopa_pack(payload, gate, ids)
    rx, rgate, rmask = pk.hopa_unpack(buf, lay)

    for name, plane in (("payload", rx), ("gate", rgate), ("id", rmask)):
        assert not _aliases(plane, buf), (
            "rows=%d dtype=%s:%s 面仍与并包缓冲共享存储 —— "
            "调用方 resize(0) 后它会悬空" % (rows, dtype, name)
        )


@pytest.mark.parametrize("rows", ROW_COUNTS)
def test_hopa_unpack_never_aliases_bitmask_format(rows):
    """位掩码线格式(id 面是 1-D)走的是另一条切片,同样不许别名。"""
    H, slots = JUDGMENT["hidden"], JUDGMENT["slots"]
    payload = torch.zeros(rows, H, dtype=torch.bfloat16)
    gate = torch.zeros(rows, slots, dtype=torch.bfloat16)
    ids = torch.zeros(rows, dtype=torch.int64)

    buf, lay = pk.hopa_pack(payload, gate, ids)
    assert lay.id_1d, "这一组本该走 1-D id 面"
    for plane in pk.hopa_unpack(buf, lay):
        assert not _aliases(plane, buf), "rows=%d:位掩码格式下解包平面别名了缓冲" % rows


@pytest.mark.parametrize("pairs", ROW_COUNTS)
def test_hopb_unpack_never_aliases_buffer(pairs):
    slot = torch.zeros(pairs, dtype=torch.int64)
    gate = torch.zeros(pairs, dtype=torch.bfloat16)
    buf = pk.hopb_pack_meta(slot, gate)
    for plane in pk.hopb_unpack_meta(buf, torch.bfloat16):
        assert not _aliases(plane, buf), "pairs=%d:Hop B 解包平面别名了缓冲" % pairs


@pytest.mark.parametrize("rows", ROW_COUNTS)
def test_planes_survive_freeing_the_buffer(rows):
    """端到端复现事故形状:解包 -> 把缓冲的存储收成 0 -> 三个平面仍可读且值正确。

    这一条才是真正对着事故写的:上面几条只查指针,这条查"还内存之后还能不能用"。
    """
    H, q = JUDGMENT["hidden"], JUDGMENT["quota"]
    payload = torch.arange(rows * H, dtype=torch.float32).reshape(rows, H).to(torch.bfloat16)
    gate = torch.full((rows, q), 0.5, dtype=torch.bfloat16)
    ids = torch.arange(rows * q, dtype=torch.int64).reshape(rows, q)

    buf, lay = pk.hopa_pack(payload, gate, ids)
    rx, rgate, rmask = pk.hopa_unpack(buf, lay)
    pk.assert_not_aliased(buf, rx, rgate, rmask)

    buf.untyped_storage().resize_(0)          # 事故现场的那一行

    assert torch.equal(rx, payload), "还内存后载荷面读出来不对"
    assert torch.equal(rgate, gate), "还内存后 gate 面读出来不对"
    assert torch.equal(rmask, ids), "还内存后 id 面读出来不对"
    # 下游第一次真用:事故里炸的就是这一句
    _ = rmask // 3


def test_assert_not_aliased_actually_catches_an_alias():
    """绊线自检:手工造一个别名平面,assert_not_aliased 必须抓到。

    没有这条,绊线一旦写错(比如把 numel 判反)就会变成永远不响的哑巴 ——
    这正是 08-22 那天 idle_watch 的死法。
    """
    buf = torch.zeros(4, 8, dtype=torch.int64)
    aliased = buf[:, :2]                       # 明确的视图,不做拷贝
    assert _aliases(aliased, buf), "构造的别名平面本身就没别名,测试无效"
    with pytest.raises(RuntimeError, match="共享存储"):
        pk.assert_not_aliased(buf, aliased)
    # 真拷贝必须放行
    pk.assert_not_aliased(buf, aliased.clone())


def test_contiguous_would_have_been_a_noop_at_one_row():
    """把"为什么 .contiguous() 不够"钉成可执行的事实。

    如果哪天 PyTorch 改了连续性判定、或有人把 _own 改回 .contiguous(),
    这条会先炸,提醒来看注释里的论证还成不成立。
    """
    buf = torch.zeros(1, 516, dtype=torch.int64)
    sliced = buf[:, :3]
    assert sliced.is_contiguous(), (
        "R==1 的列切片不再被判为连续 —— _own 的必要性论证需要重新核实"
    )
    assert sliced.contiguous().untyped_storage().data_ptr() == buf.untyped_storage().data_ptr(), (
        ".contiguous() 不再原样交回视图 —— 同上"
    )
    assert sliced.clone().untyped_storage().data_ptr() != buf.untyped_storage().data_ptr()

    two = torch.zeros(2, 516, dtype=torch.int64)[:, :3]
    assert not two.is_contiguous(), "R>=2 本该是非连续的(对照组)"
