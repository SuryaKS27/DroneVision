# tools/track_images_deep.py
import os, cv2, torch, sys, argparse, numpy as np
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.core import YAMLConfig
# This now correctly imports from your local reid.py file
from reid import ReidFeatureExtractor
from deep_vakt_tracker import DeepVAKTTracker

def save_results_to_file(tracked_objects, frame_count, file_path):
    with open(file_path, 'a') as f:
        for track in tracked_objects:
            bbox, track_id = track['bbox'], track['id']
            x, y, w, h = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
            f.write(f"{frame_count},{track_id},{x},{y},{w},{h},{track['score']:.4f},-1,-1,-1\n")

def main(args):
    cfg = YAMLConfig(args.config, resume=args.resume); checkpoint = torch.load(args.resume, map_location='cpu')
    state = checkpoint['ema']['module'] if 'ema' in checkpoint else checkpoint['model']
    cfg.model.load_state_dict(state, strict=False)

    class Model(torch.nn.Module):
        def __init__(self): super().__init__(); self.model=cfg.model.deploy(); self.postprocessor=cfg.postprocessor.deploy()
        def forward(self, i, s): return self.postprocessor(self.model(i), s)
    model = Model().to(args.device).eval()

    transforms = T.Compose([T.Resize((640, 640)), T.ToTensor()])
    os.makedirs(args.out_dir, exist_ok=True)
    
    feature_extractor = ReidFeatureExtractor(device=args.device)
    tracker = DeepVAKTTracker()
    
    image_files = sorted([f for f in os.listdir(args.img_dir) if f.lower().endswith(('.png', '.jpg'))])
    results_file_path = os.path.join(args.out_dir, "results_deep.txt")
    if os.path.exists(results_file_path): os.remove(results_file_path)
    
    for frame_count, img_name in enumerate(tqdm(image_files), 1):
        frame = cv2.imread(os.path.join(args.img_dir, img_name))
        im_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        w, h = im_pil.size; orig_size = torch.tensor([w, h])[None].to(args.device)
        im_data = transforms(im_pil)[None].to(args.device)

        with torch.no_grad():
            labels, boxes, scores = model(im_data, orig_size)
        
        det_boxes = boxes[0].cpu().numpy(); det_scores = scores[0].cpu().numpy(); det_labels = labels[0].cpu().numpy().astype(int)
        
        image_crops = [frame[int(b[1]):int(b[3]), int(b[0]):int(b[2])] for b in det_boxes]
        if not image_crops:
            embeddings = []
        else:
            embeddings = feature_extractor(image_crops).cpu().numpy()
        
        tracked_objects = tracker.update(det_boxes, det_labels, det_scores, embeddings)
        save_results_to_file(tracked_objects, frame_count, results_file_path)

    print(f"\n✅ Deep tracking complete. Results saved to {results_file_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('-c', '--config', required=True); parser.add_argument('-r', '--resume', required=True)
    parser.add_argument('--img-dir', required=True); parser.add_argument('--out-dir', default="outputs_deep_tracked")
    parser.add_argument('-d', '--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args(); main(args)
