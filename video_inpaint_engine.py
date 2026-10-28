import sys
import os
import argparse
import platform
import subprocess
import time
import gc
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
from PIL import Image
import onnxruntime as ort

# Reconfigure stdout/stderr for utf-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

def get_base_dir():
    if getattr(sys, 'frozen', False):
        exe_dir = os.path.dirname(sys.executable)
        return os.path.dirname(exe_dir)
    return os.path.dirname(os.path.abspath(__file__))

def get_ext_root_dir():
    base = get_base_dir()
    return os.path.dirname(base)

def find_ffmpeg():
    ext_root = get_ext_root_dir()
    is_win = sys.platform.startswith("win")
    bin_sub = "win" if is_win else "mac"
    exe_name = "ffmpeg.exe" if is_win else "ffmpeg"
    
    candidates = [
        os.path.join(ext_root, "bin", bin_sub, exe_name),
        os.path.join(get_base_dir(), "bin", bin_sub, exe_name),
        os.path.join(os.path.dirname(get_base_dir()), "bin", bin_sub, exe_name),
        "ffmpeg"
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return "ffmpeg"

def resolve_model(filename):
    base_dir = get_base_dir()
    ext_root = get_ext_root_dir()
    candidates = [
        os.path.join(base_dir, filename),
        os.path.join(base_dir, "engine", filename),
        os.path.join(ext_root, "engine", filename),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

class SmartMaskTracker:
    """
    Robust Smart Mask Auto-Tracker.
    Maintains exact artist mask geometry, tracking across video frames using:
    1. Lucas-Kanade Pyramidal Optical Flow on features strictly inside mask
    2. Forward-Backward consistency check to eliminate background drift
    3. RANSAC Affine Estimation (Translation, Rotation, Scale)
    4. Bounding Box Anchor (TrackerMIL) fallback for textureless regions
    Never replaces user mask with full-frame auto-segmentation.
    """
    def __init__(self, init_frame, init_mask, dilate_px=0):
        self.h, self.w = init_frame.shape[:2]

        if dilate_px > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
            self.curr_mask = cv2.dilate(init_mask, k, iterations=1)
        else:
            self.curr_mask = init_mask.copy()

        self.initial_area = max(1, np.count_nonzero(self.curr_mask))
        self.prev_gray = cv2.cvtColor(init_frame, cv2.COLOR_BGR2GRAY)

        y_idx, x_idx = self.curr_mask.nonzero()
        if len(x_idx) > 0:
            self.has_object = True
            bx = int(x_idx.min())
            by = int(y_idx.min())
            bw = max(10, int(x_idx.max() - bx))
            bh = max(10, int(y_idx.max() - by))
            self.prev_bbox = (bx, by, bw, bh)
            self.prev_center = (bx + bw / 2.0, by + bh / 2.0)

            try:
                self.tracker = cv2.TrackerMIL_create()
                self.tracker.init(init_frame, self.prev_bbox)
            except Exception:
                self.tracker = None
        else:
            self.has_object = False
            self.tracker = None
            self.prev_bbox = (0, 0, 0, 0)
            self.prev_center = (0.0, 0.0)

        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

    def track_next(self, curr_frame):
        if not self.has_object or np.count_nonzero(self.curr_mask) == 0:
            return np.zeros((self.h, self.w), dtype=np.uint8)

        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # 1. Update anchor tracker for translation reference
        tracker_dx, tracker_dy = 0.0, 0.0
        if self.tracker is not None:
            try:
                ok_track, box = self.tracker.update(curr_frame)
                if ok_track:
                    cur_center = (box[0] + box[2] / 2.0, box[1] + box[3] / 2.0)
                    tracker_dx = cur_center[0] - self.prev_center[0]
                    tracker_dy = cur_center[1] - self.prev_center[1]
                    self.prev_center = cur_center
                    self.prev_bbox = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
            except Exception:
                pass

        # 2. Extract feature points strictly inside current mask
        pts_prev = cv2.goodFeaturesToTrack(
            self.prev_gray,
            maxCorners=100,
            qualityLevel=0.01,
            minDistance=5,
            mask=self.curr_mask
        )

        mask_updated = False
        if pts_prev is not None and len(pts_prev) >= 4:
            # Forward LK
            pts_curr, s_fwd, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, curr_gray, pts_prev, None, **self.lk_params
            )
            # Backward LK for consistency check
            pts_back, s_back, _ = cv2.calcOpticalFlowPyrLK(
                curr_gray, self.prev_gray, pts_curr, None, **self.lk_params
            )

            fb_dist = abs(pts_prev - pts_back).reshape(-1, 2).max(-1)
            valid = (s_fwd.reshape(-1) == 1) & (s_back.reshape(-1) == 1) & (fb_dist < 1.0)

            good_prev = pts_prev[valid]
            good_curr = pts_curr[valid]

            if len(good_prev) >= 4:
                M, inliers = cv2.estimateAffinePartial2D(
                    good_prev, good_curr, method=cv2.RANSAC, ransacReprojThreshold=3.0
                )
                if M is not None:
                    warped = cv2.warpAffine(self.curr_mask, M, (self.w, self.h), flags=cv2.INTER_NEAREST)
                    warped_area = np.count_nonzero(warped)
                    # Area bounds sanity check: ensure mask does not explode or collapse
                    if 0.25 * self.initial_area <= warped_area <= 3.5 * self.initial_area:
                        self.curr_mask = warped
                        mask_updated = True
            elif len(good_prev) > 0:
                shift = np.median(good_curr - good_prev, axis=0)[0]
                M = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
                self.curr_mask = cv2.warpAffine(self.curr_mask, M, (self.w, self.h), flags=cv2.INTER_NEAREST)
                mask_updated = True

        if not mask_updated:
            # Fallback to tracker displacement
            M = np.float32([[1, 0, tracker_dx], [0, 1, tracker_dy]])
            self.curr_mask = cv2.warpAffine(self.curr_mask, M, (self.w, self.h), flags=cv2.INTER_NEAREST)

        self.prev_gray = curr_gray
        return self.curr_mask

def inpaint_crop(frame, mask_bin, lama_session):
    """
    Crops surrounding context around the tracked object, executes LaMa Fourier
    inpainting, and composites back with feathered edge blending.
    Preserves 100% native resolution and sharpness on untouched areas.
    """
    h_f, w_f = frame.shape[:2]
    y_idx, x_idx = mask_bin.nonzero()
    if len(x_idx) == 0 or len(y_idx) == 0:
        return frame

    bx0, bx1 = int(x_idx.min()), int(x_idx.max())
    by0, by1 = int(y_idx.min()), int(y_idx.max())
    bw, bh = bx1 - bx0, by1 - by0

    # Add 35% margin around tracked bounding box for surrounding spatial context
    pad_x = max(30, int(bw * 0.35))
    pad_y = max(30, int(bh * 0.35))
    x0 = max(0, bx0 - pad_x)
    y0 = max(0, by0 - pad_y)
    x1 = min(w_f, bx1 + pad_x)
    y1 = min(h_f, by1 + pad_y)

    crop_frame = frame[y0:y1, x0:x1]
    crop_mask = mask_bin[y0:y1, x0:x1]

    if crop_frame.size == 0 or crop_mask.size == 0 or np.count_nonzero(crop_mask) == 0:
        return frame

    orig_h, orig_w = crop_frame.shape[:2]
    img_rgb = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (512, 512), interpolation=cv2.INTER_LANCZOS4)
    mask_resized = cv2.resize(crop_mask, (512, 512), interpolation=cv2.INTER_NEAREST)

    img_input = (img_resized.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
    mask_input = (mask_resized.astype(np.float32) / 255.0)[None, None, ...]
    img_input = img_input * (1.0 - mask_input)

    inputs = {
        lama_session.get_inputs()[0].name: img_input,
        lama_session.get_inputs()[1].name: mask_input
    }
    ort_outs = lama_session.run(None, inputs)
    result = ort_outs[0][0].transpose(1, 2, 0)
    result = np.clip(result, 0, 255).astype(np.uint8)
    result_full = cv2.resize(result, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)

    # Feather edge blending for seamless composite
    feather_ksize = max(7, (int(max(orig_h, orig_w) * 0.02) * 2) | 1)
    mask_f = (crop_mask > 127).astype(np.float32)
    mask_f = cv2.GaussianBlur(mask_f, (feather_ksize, feather_ksize), 0)[..., None]

    comp_crop = (result_full * mask_f + img_rgb * (1.0 - mask_f)).astype(np.uint8)

    out_frame = frame.copy()
    out_frame[y0:y1, x0:x1] = cv2.cvtColor(comp_crop, cv2.COLOR_RGB2BGR)
    return out_frame

def process_video(input_video_path, mask_path, output_video_path, mode="moving", use_gpu=False, start_time=0.0, start_frame=0, dilate_px=0):
    temp_raw_video = output_video_path + ".temp_raw.mp4"
    cap = None
    out = None

    try:
        # 1. Hardware Provider Selection
        available = ort.get_available_providers()
        providers = []
        if use_gpu:
            for p in ['CUDAExecutionProvider', 'DmlExecutionProvider', 'CoreMLExecutionProvider']:
                if p in available:
                    providers.append(p)
        if 'CPUExecutionProvider' in available:
            providers.append('CPUExecutionProvider')
        if not providers:
            providers = ['CPUExecutionProvider']

        # 2. Resolve LaMa Inpainting Model
        lama_model_path = resolve_model("lama_fp32.onnx")
        if not lama_model_path:
            raise FileNotFoundError("Cannot find LaMa model (lama_fp32.onnx).")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = os.cpu_count() or 4
        lama_session = ort.InferenceSession(lama_model_path, sess_options=sess_options, providers=providers)

        # 3. Read Mask & Video
        mask_init = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_init is None:
            raise FileNotFoundError(f"Cannot read mask image: {mask_path}")
        _, mask_init = cv2.threshold(mask_init, 127, 255, cv2.THRESH_BINARY)

        if np.count_nonzero(mask_init) == 0:
            raise ValueError("Provided mask is empty. Please brush over the object to remove.")

        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {input_video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = 1

        if mask_init.shape[:2] != (height, width):
            mask_init = cv2.resize(mask_init, (width, height), interpolation=cv2.INTER_NEAREST)

        # Determine start frame index
        init_frame_idx = start_frame
        if init_frame_idx == 0 and start_time > 0.0:
            init_frame_idx = int(round(start_time * fps))
        init_frame_idx = max(0, min(total_frames - 1, init_frame_idx))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_raw_video, fourcc, fps, (width, height))

        print(f"PROGRESS: 5% | Initializing Smart Tracker (Mode: {mode}, Start Frame: {init_frame_idx})...", flush=True)

        # Case A: Static Mask Mode (Locked Watermark / Stationary Object)
        if mode == "static":
            k_static = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, dilate_px), max(3, dilate_px))) if dilate_px > 0 else None
            static_mask = cv2.dilate(mask_init, k_static, iterations=1) if k_static is not None else mask_init

            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                final_frame = inpaint_crop(frame, static_mask, lama_session)
                out.write(final_frame)
                frame_idx += 1
                if total_frames > 0 and (frame_idx % 5 == 0 or frame_idx == total_frames):
                    prog = int(5 + (frame_idx / total_frames) * 85)
                    print(f"PROGRESS: {prog}% | Inpainting static object frame {frame_idx}/{total_frames}...", flush=True)

        # Case B: Smart Mask Auto-Track (Moving Object)
        else:
            # If mask was drawn at frame 0, streamingly track and inpaint in a single pass (O(1) memory)
            if init_frame_idx == 0:
                ret, first_frame = cap.read()
                if not ret:
                    raise IOError("Failed to read initial frame from video.")

                tracker = SmartMaskTracker(first_frame, mask_init, dilate_px=dilate_px)
                curr_mask = tracker.curr_mask.copy()

                final_frame = inpaint_crop(first_frame, curr_mask, lama_session)
                out.write(final_frame)

                frame_idx = 1
                if total_frames > 0:
                    prog = int(5 + (frame_idx / total_frames) * 85)
                    print(f"PROGRESS: {prog}% | Auto-tracking & inpainting frame {frame_idx}/{total_frames}...", flush=True)

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    curr_mask = tracker.track_next(frame)
                    final_frame = inpaint_crop(frame, curr_mask, lama_session)
                    out.write(final_frame)

                    frame_idx += 1
                    if total_frames > 0 and (frame_idx % 5 == 0 or frame_idx == total_frames):
                        prog = int(5 + (frame_idx / total_frames) * 85)
                        print(f"PROGRESS: {prog}% | Auto-tracking & inpainting frame {frame_idx}/{total_frames}...", flush=True)

            # If mask was drawn on scrubbed frame K > 0: Low-RAM 2-Pass Architecture
            else:
                print(f"PROGRESS: 6% | Buffering pre-roll frames for backward tracking (0 to {init_frame_idx})...", flush=True)
                preroll_frames = []
                for i in range(init_frame_idx + 1):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    preroll_frames.append(frame)

                actual_buffered = len(preroll_frames)
                if actual_buffered == 0:
                    raise IOError("Failed to read frames from video.")

                init_frame_idx = min(init_frame_idx, actual_buffered - 1)
                all_masks_packed = [None] * actual_buffered

                # Step 1: Backward tracking: K -> 0
                bwd_tracker = SmartMaskTracker(preroll_frames[init_frame_idx], mask_init, dilate_px=dilate_px)
                all_masks_packed[init_frame_idx] = np.packbits(bwd_tracker.curr_mask > 0)

                for i in range(init_frame_idx - 1, -1, -1):
                    m_bwd = bwd_tracker.track_next(preroll_frames[i])
                    all_masks_packed[i] = np.packbits(m_bwd > 0)
                    if (init_frame_idx - i) % 5 == 0 or i == 0:
                        prog = int(6 + ((init_frame_idx - i) / max(1, init_frame_idx)) * 10)
                        print(f"PROGRESS: {prog}% | Tracking backwards frame {init_frame_idx - i}/{init_frame_idx}...", flush=True)

                # Keep anchor frame for forward tracker, immediately free the rest of preroll to reclaim RAM
                anchor_frame = preroll_frames[init_frame_idx]
                del preroll_frames
                gc.collect()

                # Step 2: Forward tracking: K -> N-1 (Streaming sequentially from cap, 1 frame in memory)
                fwd_tracker = SmartMaskTracker(anchor_frame, mask_init, dilate_px=dilate_px)
                del anchor_frame
                gc.collect()

                curr_track_idx = init_frame_idx + 1
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    m_fwd = fwd_tracker.track_next(frame)
                    all_masks_packed.append(np.packbits(m_fwd > 0))
                    if curr_track_idx % 5 == 0 or curr_track_idx == total_frames - 1:
                        prog = int(16 + (curr_track_idx / total_frames) * 12)
                        print(f"PROGRESS: {prog}% | Tracking forward frame {curr_track_idx}/{total_frames}...", flush=True)
                    curr_track_idx += 1

                cap.release()

                # Step 3: Pass 2 Sequential Streaming Inpaint (Reopen video, 1 frame in RAM)
                print("PROGRESS: 30% | Applying AI inpainting across tracked frames...", flush=True)
                cap = cv2.VideoCapture(input_video_path)
                num_tracked = len(all_masks_packed)
                inpaint_idx = 0

                while cap.isOpened() and inpaint_idx < num_tracked:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    packed = all_masks_packed[inpaint_idx]
                    if packed is not None:
                        m = np.unpackbits(packed).reshape(height, width) * 255
                    else:
                        m = mask_init

                    final_frame = inpaint_crop(frame, m, lama_session)
                    out.write(final_frame)
                    inpaint_idx += 1

                    if inpaint_idx % 5 == 0 or inpaint_idx == num_tracked:
                        prog = int(30 + (inpaint_idx / num_tracked) * 60)
                        print(f"PROGRESS: {prog}% | Inpainting frame {inpaint_idx}/{num_tracked}...", flush=True)

                # Free packed masks
                del all_masks_packed
                gc.collect()

        cap.release()
        out.release()
        cap = None
        out = None

        # 4. Post-process with FFmpeg for H.264 (yuv420p) compatibility and original audio preservation
        print("PROGRESS: 92% | Encoding clean H.264 video with audio...", flush=True)
        ffmpeg_bin = find_ffmpeg()
        ffmpeg_success = False

        # Always ensure standard MP4 container for H.264/AAC compatibility
        actual_output_path = output_video_path
        base_out, out_ext = os.path.splitext(output_video_path)
        if out_ext.lower() != ".mp4":
            actual_output_path = base_out + ".mp4"

        if ffmpeg_bin:
            try:
                cmd = [
                    ffmpeg_bin, "-y",
                    "-i", temp_raw_video,
                    "-i", input_video_path,
                    "-map", "0:v:0",
                    "-map", "1:a?",
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    "-preset", "fast",
                    "-crf", "18",
                    "-c:a", "aac",
                    "-b:a", "192k",
                    "-movflags", "+faststart",
                    "-shortest",
                    actual_output_path
                ]
                res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if res.returncode == 0 and os.path.exists(actual_output_path) and os.path.getsize(actual_output_path) > 1000:
                    ffmpeg_success = True
                    if actual_output_path != output_video_path and os.path.exists(output_video_path):
                        try:
                            os.remove(output_video_path)
                        except Exception:
                            pass
            except Exception as ff_err:
                print(f"FFmpeg warning: {ff_err}", file=sys.stderr, flush=True)

        if not ffmpeg_success:
            if os.path.exists(output_video_path):
                try:
                    os.remove(output_video_path)
                except Exception:
                    pass
            os.rename(temp_raw_video, output_video_path)

        print("PROGRESS: 100% | Complete!", flush=True)
        print("SUCCESS", flush=True)

    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        if out is not None:
            try:
                out.release()
            except Exception:
                pass
        # Always guarantee temporary raw video is purged
        if os.path.exists(temp_raw_video):
            try:
                os.remove(temp_raw_video)
            except Exception:
                pass

def main():
    parser = argparse.ArgumentParser(description="Asset Station Smart Mask & Auto-Track Video Inpainting Engine")
    parser.add_argument("input_video", help="Path to input video (.mp4, .mov, .webm)")
    parser.add_argument("mask", help="Path to mask image (.png)")
    parser.add_argument("output_video", help="Path to save cleaned video (.mp4)")
    parser.add_argument("--mode", default="moving", choices=["moving", "static"], help="Tracking mode: moving (Smart Auto-Track) or static")
    parser.add_argument("--start-time", type=float, default=0.0, help="Timestamp (seconds) where initial mask was painted")
    parser.add_argument("--start-frame", type=int, default=0, help="Frame index where initial mask was painted")
    parser.add_argument("--dilate", type=int, default=0, help="Mask boundary dilation in pixels")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU acceleration")

    args = parser.parse_args()
    try:
        process_video(
            args.input_video,
            args.mask,
            args.output_video,
            mode=args.mode,
            use_gpu=args.gpu,
            start_time=args.start_time,
            start_frame=args.start_frame,
            dilate_px=args.dilate
        )
    except Exception as e:
        print(f"ERROR: {str(e)}", file=sys.stderr, flush=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
