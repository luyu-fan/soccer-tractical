"""
构建HOG特征库
"""
# move the script to root dir when running

import cv2, os
import numpy as np
import pickle

import lib.interprocess.hog as hog


def picking_samples():
    root = r"C:\Users\luyu-fan\Desktop\mdp\data-store\datasets\mot-datasets\selected-datasets\MCC4KSoccer\images\test\720p\IVCEU9_22\img"
    for img_name in os.listdir(root):
        img = cv2.imread(os.path.join(root, img_name))
        cv2.imshow("picking", img)
        cv2.waitKey(1)
        _ = input()

def generate_ball_hog_bank():
    """
    生成HOG特征库
    """
    root = r"C:\Users\luyu-fan\Desktop\pos_samples"
    hog_features = []
    for img_name in os.listdir(root):
        img = cv2.imread(os.path.join(root, img_name))
        hist = hog.HOGProcess.compute(img).reshape((-1, ))
        hog_features.append(hist)
    return np.array(hog_features)

def verify():

    anchor = cv2.imread("ball.jpg")
    anchor_hist = hog.HOGProcess.compute(anchor).reshape((-1, ))
    # print(hist.shape, hist)

    pos = cv2.imread("pos.jpg")
    pos_hist = hog.HOGProcess.compute(pos).reshape((-1, ))
    pos2 = cv2.imread("pos2.jpg")
    pos2_hist = hog.HOGProcess.compute(pos2).reshape((-1, ))
    pos3 = cv2.imread("pos3.jpg")
    pos3_hist = hog.HOGProcess.compute(pos3).reshape((-1, ))
    pos4 = cv2.imread("pos4.jpg")
    pos4_hist = hog.HOGProcess.compute(pos4).reshape((-1, ))

    neg = cv2.imread("neg.jpg")
    neg_hist = hog.HOGProcess.compute(neg).reshape((-1, ))
    neg2 = cv2.imread("neg2.png")
    neg2_hist = hog.HOGProcess.compute(neg2).reshape((-1, ))
    neg3 = cv2.imread("neg3.jpg")
    neg3_hist = hog.HOGProcess.compute(neg3).reshape((-1, ))

    # print(anchor_hist.shape, anchor_hist)
    # print(pos_hist.shape, pos_hist)
    # print(neg_hist.shape, pos_hist)

    distance1 = np.linalg.norm(anchor_hist - pos_hist)
    distance2 = np.linalg.norm(anchor_hist - neg_hist)

    distance3 = np.linalg.norm(anchor_hist - pos2_hist)
    distance4 = np.linalg.norm(anchor_hist - neg2_hist)
    distance5 = np.linalg.norm(anchor_hist - pos3_hist)
    distance6 = np.linalg.norm(anchor_hist - pos4_hist)
    distance7 = np.linalg.norm(anchor_hist - neg3_hist)

    print(distance1, "+")
    print(distance2, "-")
    print(distance3, "+")
    print(distance4, "-")
    print(distance5, "+")
    print(distance6, "+")
    print(distance7, "-")

    # 在当前的设置下 28可以作为阈值


if __name__ == "__main__":


    hog_bank = generate_ball_hog_bank()

    with open("hog_bank.pkl", mode="wb") as f:
        pickle.dump(hog_bank, f)

    pos1 = cv2.imread("pos.jpg")
    pos1_hist = hog.HOGProcess.compute(pos1).reshape((1, -1))
    pos2 = cv2.imread("pos2.jpg")
    pos2_hist = hog.HOGProcess.compute(pos2).reshape((1, -1))
    pos3 = cv2.imread("pos3.jpg")
    pos3_hist = hog.HOGProcess.compute(pos3).reshape((1, -1))
    pos4 = cv2.imread("pos4.jpg")
    pos4_hist = hog.HOGProcess.compute(pos4).reshape((1, -1))

    neg1 = cv2.imread("neg.jpg")
    neg1_hist = hog.HOGProcess.compute(neg1).reshape((1, -1))
    neg2 = cv2.imread("neg2.png")
    neg2_hist = hog.HOGProcess.compute(neg2).reshape((1, -1))
    neg3 = cv2.imread("neg3.jpg")
    neg3_hist = hog.HOGProcess.compute(neg3).reshape((1, -1))

    dists = np.linalg.norm(hog_bank - pos1_hist, axis=1)
    print(dists.mean())
    dists = np.linalg.norm(hog_bank - pos2_hist, axis=1)
    print(dists.mean())
    dists = np.linalg.norm(hog_bank - pos3_hist, axis=1)
    print(dists.mean())
    dists = np.linalg.norm(hog_bank - pos4_hist, axis=1)
    print(dists.mean())
    print("----------")
    dists = np.linalg.norm(hog_bank - neg1_hist, axis=1)
    print(dists.mean())
    dists = np.linalg.norm(hog_bank - neg2_hist, axis=1)
    print(dists.mean())
    dists = np.linalg.norm(hog_bank - neg3_hist, axis=1)
    print(dists.mean())

    
