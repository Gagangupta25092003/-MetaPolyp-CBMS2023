from predict import predict_single, load_model, mask_parse
import os
import cv2

def novelty(img_size, model_path, route):
    model = load_model(model_path=model_path)
    list_of_datasets = os.listdir(route)
    print(list_of_datasets)
    for datasets in list_of_datasets:
        print(datasets, ":")
        if os.path.exists(os.path.join(route,datasets)) and os.path.exists(os.path.join(f'{os.path.join(route, datasets)}/images')):
            dir = os.listdir(f'{os.path.join(route, datasets)}/images')
            for image in dir:                
                image_path = f'{os.path.join(route, datasets)}/images/{image}'
                print("jhsdfhksjnf", image_path)
                print()
                predict_path = f'{os.path.join(route, datasets)}/predict/{image}'
                predicted_mask = predict_single(model, image_path)
                out = mask_parse(predicted_mask)
                cv2.imwrite(predict_path, out)
    




if __name__ == "__main__":

    img_size = 256
    BATCH_SIZE = 1
    SEED = 1024
    save_path = "pretrained_model.h5"
    route_data = "/home/skycoder/Desktop/Meta-polyp/metapolyp-cbms2023/TestDataset/TestDataset/"
    path_to_test_dataset = "/home/skycoder/Desktop/Meta-polyp/metapolyp-cbms2023/TestDataset/TestDataset/"
    novelty(img_size, save_path, route_data)