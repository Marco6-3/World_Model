"""Simulator-independent geometry, observation schema and temporal sampling."""
import numpy as np

ROOT = "/World/Bag"
COLLAR_Z = 0.125
LIP_HALF = 0.007
OPEN = 0.085
CLOSED = 0.012
FLOOR = 0.014
OCCLUDE_Z = 0.065
SIZE = np.array([[0.036, 0.042, 0.024], [0.028, 0.030, 0.042]])
SLOTS = np.array([[-0.048, 0., FLOOR + SIZE[0, 2]/2], [0.048, 0., FLOOR + SIZE[1, 2]/2]])
STAGING = np.array([[-0.048, 0.22, 0.075 + SIZE[0, 2]/2], [0.048, 0.22, 0.075 + SIZE[1, 2]/2]])
HANDLE_OFFSET = np.array([-0.048, -0.100, 0.050])
DYNAMICS = {"relaxed": (0., .22), "elastic_slow": (1.2, .18), "elastic_fast": (6., .10)}
CONTROL_DT = 4 / 120
OBS_DIM = 39
ACTION_DIM = 8
HISTORY = 64
HORIZON = 24
THRESHOLD = 0.004


def history_indices(t):
    # 60 spaced tokens + four consecutive tokens; 444 observations inclusive.
    offsets = np.r_[np.rint(np.linspace(-443, -4, 60)).astype(int), np.arange(-3, 1)]
    return np.maximum(0, t + offsets).astype(int)


def slot_pos(item, slot_id):
    return np.array([SLOTS[int(slot_id), 0], 0., FLOOR + SIZE[int(item), 2]/2])


def visible_contents(poses, occupancy):
    """Mask position AND both slot bits; mask location never discloses the occupied side."""
    result=[]
    hidden=[]
    for pose in poses:
        hide=bool(pose[2]<OCCLUDE_Z and abs(pose[1])<.08 and abs(pose[0])<.095)
        hidden.append(hide)
        result.extend(np.zeros(3) if hide else pose[:3]);result.append(float(not hide))
    mask=np.zeros(2) if any(hidden) else np.ones(2)
    result.extend(np.asarray(occupancy)*mask);result.extend(mask)
    return np.asarray(result,dtype=np.float32)


def effective_clearance(horizontal, centre, tip, item):
    """3D lower-bound margin: vertical separation can clear a blocked XY projection."""
    radius=float(np.linalg.norm(SIZE[item]/2))
    object_vertical=max(centre[2]-radius-(COLLAR_Z+.009),COLLAR_Z-.009-centre[2]-radius)
    gripper_vertical=max(tip[2]-.012-(COLLAR_Z+.009),COLLAR_Z-.009-tip[2]-.012)
    return max(float(horizontal),min(object_vertical,gripper_vertical))


def measure_trace(frames, stages, learned, attached, order, slip_tick):
    """Measure physical entry and recovery; never credit pickup or the other item."""
    second=1
    positions=frames[:,8+3*second:11+3*second]
    entry=(stages==1)&learned&(np.abs(positions[:,1])<.05)
    entry &= (positions[:,2]-SIZE[second,2]/2<COLLAR_Z+.005)
    entry &= (positions[:,2]+SIZE[second,2]/2>COLLAR_Z-.025)
    indices=np.flatnonzero(entry)
    slot=int(positions[indices[0],0]>0) if len(indices) else None
    withdraw=reopen=None
    if slip_tick is not None:
        first=0
        bottom=frames[:,10+3*first]-SIZE[first,2]/2
        collapsed=False
        for i in range(int(slip_tick),len(frames)):
            if stages[i]!=0:break
            collapsed |= frames[i,1]<.070
            if withdraw is None and bottom[i]>COLLAR_Z+.050:
                withdraw=i+1
            if withdraw is not None and collapsed and attached[i] and frames[i,1]>.095:
                reopen=i+1;break
    return dict(second_slot=slot,second_slot_correct=bool(slot==1-int(order)),
                withdraw_tick=withdraw,reopen_tick=reopen,withdraw_reopen=bool(withdraw is not None and reopen is not None))

