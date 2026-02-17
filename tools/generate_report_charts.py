# -*- coding: utf-8 -*-
import json
import csv
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 设置绘图风格
plt.style.use('default')
# 设置字体以支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'PingFang SC', 'Heiti TC', 'Microsoft YaHei', 'Arial'] 
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题 

def load_summary(path):
    with open(path / 'profiler' / 'summary_rank0.json', 'r') as f:
        return json.load(f)

def load_comm_breakdown(path):
    rs_time = 0.0
    ag_time = 0.0
    total_comm_csv = 0.0
    
    csv_path = path / 'profiler' / 'comm_op_summary_rank0.csv'
    if not csv_path.exists():
        return 0, 0, 0
        
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('is_comm') != 'True':
                continue
            t = float(row['cuda_time_total_ms'])
            total_comm_csv += t
            name = row['name']
            if 'ReduceScatter' in name:
                rs_time += t
            elif 'AllGather' in name:
                ag_time += t
                
    return total_comm_csv, rs_time, ag_time

def average_step_time(summary):
    steps = summary.get('overlap', {}).get('per_step', [])
    if not steps:
        return 0.0
    totals = []
    for step in steps:
        compute_ms = step.get('compute_total_ms_strict', 0.0)
        comm_ms = step.get('comm_total_ms', 0.0)
        totals.append(compute_ms + comm_ms)
    return float(np.mean(totals))

def average_step_parts(summary):
    steps = summary.get('overlap', {}).get('per_step', [])
    if not steps:
        return 0.0, 0.0
    compute_vals = [s.get('compute_total_ms_strict', 0.0) for s in steps]
    comm_vals = [s.get('comm_total_ms', 0.0) for s in steps]
    return float(np.mean(compute_vals)), float(np.mean(comm_vals))

def plot_scale_sensitivity(base_dir, output_dir):
    """绘制图表 1-3：规模敏感性分析 - 通信暴露率变化与压缩投资回报"""
    
    # 三规模数据（从 JSON 提取）
    scales = ['Small\n(62M)', 'Medium\n(125M)', 'Large\n(300M+)']
    scale_labels = ['Small', 'Medium', 'Large']
    
    # 通信特征数据
    comm_total = [2958, 9215, 21122]  # 总通信时间 (ms)
    exposure_ratios = [0.734, 0.426, 0.228]  # 暴露率
    rs_times = [1165, 3526, 8013]  # ReduceScatter 耗时 (ms)
    
    # 计算压缩收益（50% 压缩的理论省时）
    compression_gains = [rs * exp * 0.5 for rs, exp in zip(rs_times, exposure_ratios)]  # ΔT_exposed
    
    # 创建双子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图：通信暴露率趋势
    ax1.plot(range(len(scales)), [x*100 for x in exposure_ratios], 'o-', 
             linewidth=3, markersize=12, color='#E91E63', label='通信暴露率')
    ax1.fill_between(range(len(scales)), [x*100 for x in exposure_ratios], alpha=0.15, color='#E91E63')
    ax1.set_xticks(range(len(scales)))
    ax1.set_xticklabels(scales)
    ax1.set_ylabel('通信暴露率 (%)', fontsize=12, fontweight='bold')
    ax1.set_title('通信暴露率随模型规模的变化', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim(0, 100)
    
    # 标注数值
    for i, (scale, exp) in enumerate(zip(scale_labels, exposure_ratios)):
        ax1.text(i, exp*100 + 3, '%.1f%%' % (exp*100), ha='center', fontweight='bold', fontsize=11)
    
    # 右图：压缩投资回报率（相对步时增益）
    step_times = [sum(comm_total[i:i+1]) * (exposure_ratios[i] + 0.5) for i in range(len(scales))]  # 近似步时
    gains_pct = [(compression_gains[i] / step_times[i] * 100) if step_times[i] > 0 else 0 
                 for i in range(len(scales))]
    
    colors_gain = ['#4CAF50', '#FF9800', '#9E9E9E']
    bars = ax2.bar(range(len(scales)), gains_pct, color=colors_gain, alpha=0.85, width=0.6)
    ax2.set_xticks(range(len(scales)))
    ax2.set_xticklabels(scales)
    ax2.set_ylabel('相对步时增益 (%)', fontsize=12, fontweight='bold')
    ax2.set_title('ReduceScatter 50% 压缩的实际收益', fontsize=13, fontweight='bold')
    ax2.set_ylim(0, 40)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 添加数值标签
    for bar, gain in zip(bars, gains_pct):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1.5,
                '%.0f%%' % gain, ha='center', fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    fig.savefig(output_dir / 'fig5_scale_sensitivity.png', dpi=300)
    print("Saved %s" % (output_dir / 'fig5_scale_sensitivity.png'))
    plt.close(fig)


def main():
    base_dir = Path(__file__).parent.parent
    double_path = base_dir / 'usefuldata/step1-20260216-122021-FP32-DUAL-medium'
    single_path = base_dir / 'usefuldata/step1-20260216-124501-FP32-SINGLE-medium'
    output_dir = base_dir / 'report/assets'
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载数据
    single_summary = load_summary(single_path)
    double_summary = load_summary(double_path)
    
    # 提取 Single 数据
    s_ov = single_summary['overlap']['overall']
    s_comm = s_ov['comm_total_ms']
    s_comp = s_ov['compute_total_ms_strict']
    s_total = s_comm + s_comp
    s_comm_ratio = (s_comm / s_total * 100) if s_total > 0 else 0

    # 提取 Double 数据
    d_ov = double_summary['overlap']['overall']
    d_comm = d_ov['comm_total_ms']
    d_comp = d_ov['compute_total_ms_strict']
    d_total = d_comm + d_comp
    d_comm_ratio = (d_comm / d_total * 100) if d_total > 0 else 0
    # 使用 strict 数据
    d_exposed_ratio = d_ov.get('comm_exposed_ratio_strict', 0) * 100

    # 提取 Double 通信分解
    d_csv_total, d_rs, d_ag = load_comm_breakdown(double_path)
    # 校准: 使用 csv 的比例应用到 summary 的总时间上，或者直接用 summary 时间
    # 这里直接使用 csv 的绝对值比例来画饼图
    d_other = d_csv_total - d_rs - d_ag
    if d_other < 0: d_other = 0

    print("Single Comm Ratio: %.2f%%" % s_comm_ratio)
    print("Double Comm Ratio: %.2f%%" % d_comm_ratio)
    print("Double Comm Breakdown: RS=%.2f, AG=%.2f, Other=%.2f" % (d_rs, d_ag, d_other))

    # --- 图表 1: 不同模型规模下单卡与双卡耗时对比 (Grouped Stacked Bar Chart) ---
    fig1, ax1 = plt.subplots(figsize=(9, 6))

    scale_labels = ['Small', 'Medium', 'Large']
    single_paths = [
        base_dir / 'usefuldata/step1-20260216-124745-FP32-SINGLE-small',
        base_dir / 'usefuldata/step1-20260216-124501-FP32-SINGLE-medium',
        base_dir / 'usefuldata/step1-20260216-123914-FP32-SINGLE-large'
    ]
    dual_paths = [
        base_dir / 'usefuldata/step1-20260216-121657-FP32-DUAL-small',
        base_dir / 'usefuldata/step1-20260216-122021-FP32-DUAL-medium',
        base_dir / 'usefuldata/step1-20260216-122425-FP32-DUAL-large'
    ]

    single_compute = []
    single_comm = []
    dual_compute = []
    dual_comm = []
    for s_path, d_path in zip(single_paths, dual_paths):
        s_comp, s_comm = average_step_parts(load_summary(s_path))
        d_comp, d_comm = average_step_parts(load_summary(d_path))
        single_compute.append(s_comp)
        single_comm.append(s_comm)
        dual_compute.append(d_comp)
        dual_comm.append(d_comm)

    x = np.arange(len(scale_labels))
    width = 0.35

    bars_s_comp = ax1.bar(x - width/2, single_compute, width, color='#4CAF50', label='计算时间（单卡）')
    bars_s_comm = ax1.bar(x - width/2, single_comm, width, bottom=single_compute,
                          color='#81C784', label='通信时间（单卡）')
    bars_d_comp = ax1.bar(x + width/2, dual_compute, width, color='#2196F3', label='计算时间（双卡）')
    bars_d_comm = ax1.bar(x + width/2, dual_comm, width, bottom=dual_compute,
                          color='#F44336', label='通信时间（双卡）')

    ax1.set_xticks(x)
    ax1.set_xticklabels(scale_labels)
    ax1.set_xlabel('模型大小')
    ax1.set_ylabel('平均 Step 耗时 (ms)')
    ax1.set_title('不同模型规模下单卡与双卡耗时对比')
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    ax1.legend(loc='upper left', ncol=2)

    for i in range(len(scale_labels)):
        s_total = single_compute[i] + single_comm[i]
        d_total = dual_compute[i] + dual_comm[i]
        if s_total > 0:
            s_comp_pct = single_compute[i] / s_total * 100
            s_comm_pct = single_comm[i] / s_total * 100
            ax1.text(x[i] - width/2, single_compute[i] / 2,
                     '%.1f%%' % s_comp_pct, ha='center', va='center', color='white', fontsize=9)
            if single_comm[i] > 0:
                ax1.text(x[i] - width/2, single_compute[i] + single_comm[i] / 2,
                         '%.1f%%' % s_comm_pct, ha='center', va='center', color='white', fontsize=9)
        if d_total > 0:
            d_comp_pct = dual_compute[i] / d_total * 100
            d_comm_pct = dual_comm[i] / d_total * 100
            ax1.text(x[i] + width/2, dual_compute[i] / 2,
                     '%.1f%%' % d_comp_pct, ha='center', va='center', color='white', fontsize=9)
            ax1.text(x[i] + width/2, dual_compute[i] + dual_comm[i] / 2,
                     '%.1f%%' % d_comm_pct, ha='center', va='center', color='white', fontsize=9)

    plt.tight_layout()
    fig1.savefig(output_dir / 'fig1_comm_ratio_comparison.png', dpi=300)
    print("Saved %s" % (output_dir / 'fig1_comm_ratio_comparison.png'))

    # --- 图表 2 & 3: 通信分解与暴露分析合并 (Combined Pie Charts) ---
    fig_combined, (ax2, ax3) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 子图 1: 通信算子分解
    if d_csv_total > 0:
        sizes = [d_rs, d_ag, d_other]
        labels = ['Reduce-Scatter', 'All-Gather', '其他 (Others)']
        colors = ['#FF9800', '#2196F3', '#9E9E9E']
        explode = (0.05, 0, 0)
        
        sizes_f = []
        labels_f = []
        colors_f = []
        explode_f = []
        for s, l, c, e in zip(sizes, labels, colors, explode):
            if s > 0:
                sizes_f.append(s)
                labels_f.append(l)
                colors_f.append(c)
                explode_f.append(e)

        ax2.pie(sizes_f, explode=explode_f, labels=labels_f, colors=colors_f,
                autopct='%1.1f%%', shadow=True, startangle=140)
        ax2.set_title('通信算子耗时分解 (双卡)')

    # 子图 2: 通信暴露分析 (Pie Chart)
    # 暴露 vs 重叠
    exposed_ms = d_comm * (d_exposed_ratio / 100.0)
    covered_ms = d_comm - exposed_ms
    
    sizes_exp = [exposed_ms, covered_ms]
    labels_exp = ['暴露通信 (Exposed/Blocking)', '计算重叠 (Overlapped/Hidden)']
    colors_exp = ['#E91E63', '#8BC34A']
    explode_exp = (0.05, 0) # 突出显示暴露部分

    ax3.pie(sizes_exp, explode=explode_exp, labels=labels_exp, colors=colors_exp,
            autopct='%1.1f%%', shadow=True, startangle=140)
    ax3.set_title('通信暴露比例分析')
    
    plt.tight_layout()
    fig_combined.savefig(output_dir / 'fig2_3_combined_pie.png', dpi=300)
    print("Saved %s" % (output_dir / 'fig2_3_combined_pie.png'))

    # --- 图表 4: 平均 Step 时间分解与对比 (Average Step Time Breakdown) ---
    fig4, ax4 = plt.subplots(figsize=(6, 6))
    
    # 计算平均值
    s_steps = single_summary['overlap']['per_step']
    s_avg_compute = np.mean([s['compute_total_ms_strict'] for s in s_steps])
    
    d_steps = double_summary['overlap']['per_step']
    d_avg_compute = np.mean([s['compute_total_ms_strict'] for s in d_steps])
    d_avg_comm = np.mean([s['comm_total_ms'] for s in d_steps])
    
    labels = ['单卡 (Single-GPU)', '双卡 (Dual-GPU)']
    width = 0.5
    
    # Single 柱状图
    p1 = ax4.bar(labels[0], s_avg_compute, width, label='其他时间', color='#4CAF50', alpha=0.8)
    
    # Double 堆叠柱状图
    # 底部: 计算部分
    p2 = ax4.bar(labels[1], d_avg_compute, width, color='#2196F3', alpha=0.8)
    # 顶部: 全部通信部分
    p3 = ax4.bar(labels[1], d_avg_comm, width, bottom=d_avg_compute, label='通信总时间 (含重叠)', color='#F44336', alpha=0.9)
    
    ax4.set_ylabel('平均 Step 耗时 (ms)')
    ax4.set_title('平均 Step 耗时构成对比')
    
    # 手动构建 Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#4CAF50', alpha=0.8, label='其他时间 (单卡)'),
        Patch(facecolor='#2196F3', alpha=0.8, label='其他时间 (双卡)'),
        Patch(facecolor='#F44336', alpha=0.9, label='通信总时间 (双卡)')
    ]
    ax4.legend(handles=legend_elements, loc='upper left')
    
    ax4.grid(axis='y', linestyle='--', alpha=0.3)
    
    # 添加数值标签 (改为百分比)
    # 单卡: 其他时间占 100%
    ax4.text(labels[0], s_avg_compute/2, '100%', ha='center', va='center', color='white', fontweight='bold')
    
    # 双卡: 计算各部分占比
    d_total = d_avg_compute + d_avg_comm # 注意: 这里堆叠的高度其实是 d_avg_compute + d_avg_comm, 但这可能大于实际 Step Time (因为 overlap)
    # 但由于我们画的是堆叠图，物理高度就是两者之和。
    # 这里的百分比应该相对于 "堆叠总高度" 还是 "实际 Step Time"？
    # 通常堆叠图的百分比是指占该柱子总高度的比例。
    
    d_compute_pct = d_avg_compute / d_total * 100
    d_comm_pct = d_avg_comm / d_total * 100
    
    ax4.text(labels[1], d_avg_compute/2, f'{d_compute_pct:.1f}%', ha='center', va='center', color='white', fontweight='bold')
    ax4.text(labels[1], d_avg_compute + d_avg_comm/2, f'{d_comm_pct:.1f}%', ha='center', va='center', color='white', fontweight='bold')
    
    plt.tight_layout()
    fig4.savefig(output_dir / 'fig4_avg_step_time_breakdown.png', dpi=300)
    print(f"Saved {output_dir / 'fig4_avg_step_time_breakdown.png'}")


if __name__ == "__main__":
    main()
