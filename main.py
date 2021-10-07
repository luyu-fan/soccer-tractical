from dataprocess import prepare

if __name__ == "__main__":

    prepare.prepare_frames("BXZNP1_17.mp4")
    labels_dict = prepare.prepare_labels("BXZNP1_17.mp4")
    
