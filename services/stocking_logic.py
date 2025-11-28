import os
import json
import cv2
import torch
import numpy as np
from PIL import Image
from app.core import ML_CONTEXT


# ================= 🛠️ 辅助函数：时间格式化 =================

def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"


# ================= 🧠 提示词加载逻辑 (支持热更新) =================

def load_prompts_from_file():
    """
    每次请求时从 JSON 读取最新的 Prompt 配置
    """
    # 假设 prompts.json 在项目根目录
    # 如果你在IDE中运行，Working Directory 通常就是项目根目录
    json_path = "av_stocking_general.json"

    # 简单的默认空配置，防止文件读取失败导致崩溃
    default_t1 = {"IGNORE": [], "NOISE": [], "TARGET": []}
    default_t2 = {}

    if not os.path.exists(json_path):
        print(f"⚠️ [Warning] {json_path} not found. Using empty config.")
        return default_t1, default_t2

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # 确保获取到对应的 key
            t1 = data.get("TIER_1", default_t1)
            t2 = data.get("TIER_2", default_t2)
            return t1, t2
    except Exception as e:
        print(f"❌ [Error] Failed to read prompts.json: {e}")
        return default_t1, default_t2


def precompute_features():
    """
    加载 JSON -> 使用全局模型计算文本特征
    """
    model = ML_CONTEXT.get('model')
    tokenizer = ML_CONTEXT.get('tokenizer')
    device = ML_CONTEXT.get('device')

    if not model:
        raise RuntimeError("❌ Model not loaded in ML_CONTEXT")

    # 1. 动态加载提示词
    t1_config, t2_config = load_prompts_from_file()

    # --- Tier 1 特征计算 ---
    # 安全获取列表，防止 JSON 缺少 key
    ignores = t1_config.get("IGNORE", [])
    noises = t1_config.get("NOISE", [])
    targets = t1_config.get("TARGET", [])

    t1_prompts = ignores + noises + targets
    c_ign = len(ignores)
    c_noi = len(noises)
    c_tar = len(targets)

    if not t1_prompts:
        # 如果提示词为空，创建一个空的 dummy feature 防止报错
        t1_feats = torch.zeros((1, 1024)).to(device)  # 假设维度
    else:
        with torch.no_grad():
            toks = tokenizer(t1_prompts).to(device)
            t1_feats = model.encode_text(toks)
            t1_feats /= t1_feats.norm(dim=-1, keepdim=True)

    t1_data = {
        "features": t1_feats,
        "slices": {
            "IGNORE": slice(0, c_ign),
            "NOISE": slice(c_ign, c_ign + c_noi),
            "TARGET": slice(c_ign + c_noi, c_ign + c_noi + c_tar)
        }
    }

    # --- Tier 2 特征计算 ---
    t2_data = {}
    for name, config in t2_config.items():
        pos = config.get("pos", [])
        neg = config.get("neg", [])
        weight = config.get("weight", 1.0)  # 获取权重，默认为 1.0

        prompts = pos + neg
        if not prompts: continue

        with torch.no_grad():
            toks = tokenizer(prompts).to(device)
            feats = model.encode_text(toks)
            feats /= feats.norm(dim=-1, keepdim=True)

        t2_data[name] = {
            "features": feats,
            "pos_count": len(pos),
            "weight": weight  # 将权重存入数据包
        }

    return t1_data, t2_data


# ================= 🎯 核心分析逻辑 =================

def analyze_frame_custom(image, t1_data, t2_data):
    """
    单帧分析函数
    """
    model = ML_CONTEXT['model']
    preprocess = ML_CONTEXT['preprocess']
    device = ML_CONTEXT['device']

    # 图像预处理
    image_input = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        img_features = model.encode_image(image_input)
        img_features /= img_features.norm(dim=-1, keepdim=True)

        # === 1. Tier 1: 粗筛 ===
        # 即使 t1_prompts 为空，这里也会算出 meaningless scores，不会崩
        if t1_data["features"].shape[0] > 1:
            raw_scores = (100.0 * img_features @ t1_data["features"].T).cpu().numpy()[0]

            def get_group_score(slice_obj):
                if slice_obj.start == slice_obj.stop: return 0.0
                scores = raw_scores[slice_obj]
                scores.sort()
                top_k = scores[-3:] if len(scores) >= 3 else scores
                return np.mean(top_k)

            s_ignore = get_group_score(t1_data["slices"]["IGNORE"])
            s_noise = get_group_score(t1_data["slices"]["NOISE"])
            s_target = get_group_score(t1_data["slices"]["TARGET"])

            # 门卫判定逻辑
            if (s_target < s_ignore) or (s_target < s_noise) or (s_target < 22.0):
                return "IGNORE", float(max(s_ignore, s_noise))
        else:
            # 如果没有配置 Tier 1，默认跳过门卫（或者默认拦截，取决于你的需求，这里默认拦截）
            s_target = 0.0

        # === 2. Tier 2: 细分与权重应用 ===
        cat_scores = {}
        for name, data in t2_data.items():
            probs = (100.0 * img_features @ data["features"].T).softmax(dim=-1).cpu().numpy()[0]
            raw_score = float(sum(probs[:data["pos_count"]]))

            # 🔥 核心：应用 JSON 中配置的 weight
            weight = data["weight"]

            # 只有当原始分达到一定基准 (0.25) 且权重有效时，才进行加权
            if raw_score > 0.25 and weight != 1.0:
                final_score = raw_score * weight
            else:
                final_score = raw_score

            cat_scores[name] = final_score

        if not cat_scores:
            return "others", float(s_target / 100.0)

        # 选出最高分
        best_label = max(cat_scores, key=cat_scores.get)
        best_score = cat_scores[best_label]

        if best_score > 0.35:
            return best_label, best_score

        # 兜底
        return "others", float(s_target / 100.0)


# ================= ✂️ 时间轴合并逻辑 =================

# ================= ✂️ 优化后的时间轴合并逻辑 =================

def merge_timeline(raw_events):
    """
    两阶段合并策略：防止切分太碎
    """
    if not raw_events: return []
    # 按时间排序
    raw_events.sort(key=lambda x: x[0])

    # --- 第一阶段：基础物理合并 (同标签，时间连续) ---
    # 策略：只要间隔小于 5秒 且 标签相同，就视为物理连续
    BASE_TOLERANCE = 5

    pass1_segments = []
    curr_label = None
    start_t = 0
    last_t = 0
    score_accum = 0
    count = 0

    for t, label, score in raw_events:
        if curr_label is None:
            curr_label = label
            start_t = t
            last_t = t
            score_accum = score
            count = 1
            continue

        time_gap = t - last_t

        # 只要是同一个标签，且中间断档很小，就直接连起来
        if (label == curr_label) and (time_gap <= BASE_TOLERANCE):
            last_t = t
            score_accum += score
            count += 1
        else:
            # 结算
            pass1_segments.append({
                "start": start_t,
                "end": last_t,
                "duration": last_t - start_t,
                "label": curr_label,
                "score": score_accum / count
            })
            # 开启新段
            curr_label = label
            start_t = t
            last_t = t
            score_accum = score
            count = 1

    # 结算最后一段
    if curr_label:
        pass1_segments.append({
            "start": start_t,
            "end": last_t,
            "duration": last_t - start_t,
            "label": curr_label,
            "score": score_accum / count
        })

    # --- 第二阶段：语义缝合 (Semantic Stitching) ---
    # 解决 "碎" 的核心：如果两段 "丝袜" 中间隔了 30秒 的 "others" 或 "空窗"，
    # 我们认为这其实是同一场戏，强行合并。

    SEMANTIC_GAP_TOLERANCE = 60.0  # 🔥 核心参数：允许最大 60秒 的跨度
    MIN_FINAL_DURATION = 8.0  # 🔥 最终过滤：小于 8秒 的片段不要

    final_segments = []
    if not pass1_segments: return []

    # 取出第一个片段作为当前基准
    current_seg = pass1_segments[0]

    for next_seg in pass1_segments[1:]:
        # 计算两个片段之间的空隙
        gap = next_seg["start"] - current_seg["end"]

        is_same_label = (current_seg["label"] == next_seg["label"])
        is_close_enough = (gap <= SEMANTIC_GAP_TOLERANCE)

        if is_same_label and is_close_enough:
            # ✅ 执行合并
            # 更新结束时间
            current_seg["end"] = next_seg["end"]
            # 更新时长
            current_seg["duration"] = current_seg["end"] - current_seg["start"]
            # 更新分数 (加权平均，简单处理就取平均)
            current_seg["score"] = (current_seg["score"] + next_seg["score"]) / 2
        else:
            # ❌ 无法合并，将当前片段归档
            if current_seg["duration"] >= MIN_FINAL_DURATION:
                current_seg["time_str"] = f"{format_time(current_seg['start'])} - {format_time(current_seg['end'])}"
                final_segments.append(current_seg)

            # 切换到下一个
            current_seg = next_seg

    # 别忘了最后一个
    if current_seg["duration"] >= MIN_FINAL_DURATION:
        current_seg["time_str"] = f"{format_time(current_seg['start'])} - {format_time(current_seg['end'])}"
        final_segments.append(current_seg)

    return final_segments


# ================= 🎬 业务主入口 =================

def execute_stocking_scan(video_path: str):
    """
    业务入口：读取 JSON -> 预计算特征 -> 扫描视频 -> 合并结果
    """
    print(f"\n🎬 [StockingLogic] Start scanning: {os.path.basename(video_path)}")

    # 1. 预计算特征 (每次请求都会执行，保证 JSON 变动即时生效)
    try:
        t1_data, t2_data = precompute_features()
    except Exception as e:
        print(f"❌ Feature computation failed: {e}")
        raise e

    # 2. 视频初始化
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 硬编码策略：2秒检测一次
    CHECK_INTERVAL = 2
    step_frames = int(fps * CHECK_INTERVAL)

    raw_timeline = []

    try:
        # 使用 OpenCV 遍历
        for frame_idx in range(0, total_frames, step_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret: break

            current_sec = frame_idx / fps

            # BGR -> RGB -> PIL
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)

            # 核心识别
            label, score = analyze_frame_custom(pil_img, t1_data, t2_data)

            if label != "IGNORE":
                raw_timeline.append((current_sec, label, score))
                # 实时日志 (可选，生产环境可注释)
                # print(f"\rFound: {label} at {format_time(current_sec)} ({score:.2f})", end="")

    finally:
        cap.release()

    print(f"\n✅ [StockingLogic] Scan finished. Merging timeline...")

    # 3. 结果合并
    final_segments = merge_timeline(raw_timeline)

    return final_segments