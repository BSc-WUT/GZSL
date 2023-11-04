from pydantic import BaseModel
from typing import List

class ModelLayer(BaseModel):
    'layer_name': str
    'input_shape': str
    'output_shape': str
    'params': int | None
    'kernel_size': str | None

class Model(BaseModel):
    layers: List[ModelLayer]
    'Total params': int
    'Trainable params': int
    'Non-trainable params': int
    'Total mult-adds (Units.MEGABYTES)': float
    'Input size (MB)': float
    'Forward/backward pass size (MB)': float
    'Params size (MB)': float
    'Estimated Total Size (MB)': float

class ModelShort(BaseModel):
    layers: List[str]
    'input_shape': str
    'output_shape': str
    'Total params': int


class NetworkFlow(BaseModel):
    src_ip: str
    dst_ip: str
    src_port: str
    dst_port: str
    protocol: str
    timestamp: str
    flow_duration: str
    flow_byts_s: str
    flow_pkts_s: str
    fwd_pkts_s: str
    bwd_pkts_s: str
    tot_fwd_pkts: str
    tot_bwd_pkts: str
    totlen_fwd_pkts: str
    totlen_bwd_pkts: str
    fwd_pkt_len_max: str
    fwd_pkt_len_min: str
    fwd_pkt_len_mean: str
    fwd_pkt_len_std: str
    bwd_pkt_len_max: str
    bwd_pkt_len_min: str
    bwd_pkt_len_mean: str
    bwd_pkt_len_std: str
    pkt_len_max: str
    pkt_len_min: str
    pkt_len_mean: str
    pkt_len_std: str
    pkt_len_var: str
    fwd_header_len: str
    bwd_header_len: str
    fwd_seg_size_min: str
    fwd_act_data_pkts: str
    flow_iat_mean: str
    flow_iat_max: str
    flow_iat_min: str
    flow_iat_std: str
    fwd_iat_tot: str
    fwd_iat_max: str
    fwd_iat_min: str
    fwd_iat_mean: str
    fwd_iat_std: str
    bwd_iat_tot: str
    bwd_iat_max: str
    bwd_iat_min: str
    bwd_iat_mean: str
    bwd_iat_std: str
    fwd_psh_flags: str
    bwd_psh_flags: str
    fwd_urg_flags: str
    bwd_urg_flags: str
    fin_flag_cnt: str
    syn_flag_cnt: str
    rst_flag_cnt: str
    psh_flag_cnt: str
    ack_flag_cnt: str
    urg_flag_cnt: str
    ece_flag_cnt: str
    down_up_ratio: str
    pkt_size_avg: str
    init_fwd_win_byts: str
    init_bwd_win_byts: str
    active_max: str
    active_min: str
    active_mean: str
    active_std: str
    idle_max: str
    idle_min: str
    idle_mean: str
    idle_std: str
    fwd_byts_b_avg: str
    fwd_pkts_b_avg: str
    bwd_byts_b_avg: str
    bwd_pkts_b_avg: str
    fwd_blk_rate_avg: str
    bwd_blk_rate_avg: str
    fwd_seg_size_avg: str
    bwd_seg_size_avg: str
    cwe_flag_count: str
    subflow_fwd_pkts: str
    subflow_bwd_pkts: str
    subflow_fwd_byts: str
    subflow_bwd_byts: str
