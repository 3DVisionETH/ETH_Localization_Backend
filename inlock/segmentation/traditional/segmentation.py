import cv2
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
import logging
import easyocr
import os 
from inlock.segmentation.ml.room_segmentation import load_room_segmentation_model, generate_mask

reader = easyocr.Reader(['ch_sim','en'])
logger = logging.getLogger(__name__)

def is_approx_rect(contour, thr = 10):
    n = len(contour)
    angles = np.zeros(n)
    for i in range(n):
        pt1 = contour[i - 1][0]
        pt2 = contour[i][0]
        pt3 = contour[(i + 1) % n][0]
        
        angle = calculate_angle(pt1, pt2, pt3)
        angles[i] = angle 
        
    return np.mean((angles - 90)**2) < thr*thr

def approx_contours(gray, min_width, min_length):
    contours, hierarchy = cv2.findContours((255*gray).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    approx_contours = []
    epsilon_factor = 0.03  # You can adjust this factor

    for contour in contours:
        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        approx_contour = cv2.approxPolyDP(contour, epsilon, True)
        
        x,y = approx_contour[:,0,0], approx_contour[:,0,1]

        w,h = np.max(x)-np.min(x), np.max(y)-np.min(y)
        x0 = np.min(x),np.min(y)
        x1 = np.max(x),np.min(y)
        x2 = np.max(x),np.max(y)
        x3 = np.min(x),np.max(y)

        approx_contour = np.array([x0,x1,x2,x3])[:,None]

        criteria = is_approx_rect(approx_contour) and min(w,h) > min_width and max(w,h) > min_length
        if not criteria:
            continue
        approx_contours.append(np.array([x0,x1,x2,x3])[:,None])

    return gray, approx_contours

def calculate_angle(pt1, pt2, pt3):
    vec1 = pt1 - pt2
    vec2 = pt3 - pt2
    
    dot_product = np.dot(vec1, vec2)
    magnitude1 = np.linalg.norm(vec1)
    magnitude2 = np.linalg.norm(vec2)
    
    cos_theta = dot_product / (magnitude1 * magnitude2)
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Clip to handle numerical errors
    return np.degrees(angle)

def preprocess(img, save_fig, conf):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if conf["use_room_model"]:
        model = load_room_segmentation_model()
        gray = generate_mask(model, gray)
        gray = 1-gray
        ret,gray = cv2.threshold(gray, 0.3, 1.0, cv2.THRESH_BINARY)
    else:
        gray = 1 - gray.astype(np.float32)/255
        gray = cv2.GaussianBlur(gray, (5,5), 1.0)
        ret,gray = cv2.threshold(gray, 0.3, 1.0, cv2.THRESH_BINARY)
        edges = cv2.Canny((gray*255).astype(np.uint8), 100, 200)

    if save_fig:
        cv2.imwrite(save_fig, gray)

    return gray

def extract_contours(img, gray, save_fig, conf):
    size = min(gray.shape[0:2])
    _,contours_extracted = approx_contours(gray, min_width= size*conf["min_width"], min_length=size*conf["min_length"])

    if save_fig:
        image_with_contours = img.copy()
        cv2.drawContours(image_with_contours, contours_extracted, -1, (255, 0, 0), 2)
        cv2.imwrite(save_fig, image_with_contours)

    return contours_extracted

def extract_text(img, bbox):
    x, y, w, h = bbox
    sub = img[y-h:y+h, x-w:x+w]

    height,width,_ = sub.shape
    sub = cv2.resize(sub, (width*2, height*2))

    text = reader.readtext(sub)
    #plt.title(text)
    #plt.imshow(sub)
    #plt.show()
    return text

def extract_text_from_contours(img, contours, conf):
    extract_sf = conf["extract_text_sf"]

    res = []
    for contour in contours:
        contour = contour.reshape(4, 2)

        a = contour[:, 0]
        b = contour[:, 1]
        x0, y0 = int(min(a)), int(min(b))
        x2, y2 = int(max(a)), int(max(b))

        x , y = (x0 + x2) // 2, (y0 + y2) // 2

        w, h = (x2 - x0) // 2, (y2 - y0) // 2
        ## assume room identification is close to centre
        w = int(w*extract_sf)
        h = int(h*extract_sf)
        bbox = (x, y, w, h)
        text = extract_text(img, bbox)

        res.append([contour, text])

    return res

def plot_segmentation(img, matches, conf, output_fig: str = ""):
    ## visualise results
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = conf["font_scale"]
    font_color = (255, 255, 255)  # White color
    line_type = conf["label_thickness"]

    img = img.copy()

    for contour, text in matches:
        a = contour[:, 0]
        b = contour[:, 1]
        x0, y0 = int(min(a)), int(min(b))
        x2, y2 = int(max(a)), int(max(b))

        x , y = (x0 + x2) // 2, (y0 + y2) // 2
        location = (x, y)

        if not text:
            text = "??"
            font_color = (0, 0, 255)
        elif type(text) is list:
            _, text, score = text[0]
            font_color = (255, 0, 0)
        else:
            font_color = (255, 0, 0)

        cv2.putText(img,text,location,font, font_scale,font_color,line_type)

        cv2.rectangle(img, (x0, y0), (x2, y2), (0, 255, 0), 5)

    if output_fig != "":
        cv2.imwrite(output_fig, img)
    else:
        plt.plot()

def filter_matches(matches, conf):
    result = []
    for contour, texts in matches:
        if not texts:
            continue

        best_score = 0
        label = ""
        for _, text, score in texts:
            has_number = any((c.isnumeric() for c in text))
            criteria = has_number
            if score > best_score and criteria:
                label = text

        if label:
            result.append([contour, label])
    return result

def segment(image_path, output_path, conf):
    image = cv2.imread(image_path)
    if image is None:
        logger.error("Could not read image "+image_path)
        return 
    if not os.path.exists(output_path):
        logger.error("Could not find directory"+output_path)
        return

    output_path = Path(output_path)

    gray = preprocess(image, output_path / "room_mask.png", conf)
    contours = extract_contours(image, gray, output_path / "room_rects.png", conf)
    matches = extract_text_from_contours(image, contours, conf)
    plot_segmentation(image, matches, conf, output_path / "text_label_all.png")
    matches = filter_matches(matches, conf)
    plot_segmentation(np.zeros_like(image), matches, conf, output_path / "text_label_final.png")

default_conf = {
    "min_width": 0.01,
    "min_length": 0.03,
    "extract_text_sf": 0.8,
    "font_scale": 0.5,
    "label_thickness": 2,
    "use_room_model": True
}

if __name__ == "__main__":
    # Quick test for debugging
    # Use scripts/preprocess.py for the command line interface

    conf = default_conf

    path = "data/segmentation/gt/CAB_Floor_E.png"
    output_dir = "data/segmentation/output/CAB_E"

    segment(path, output_dir, conf)


