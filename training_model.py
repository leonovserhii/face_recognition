import os
import pickle
import sys
from icrawler.builtin import GoogleImageCrawler
import face_recognition
import cv2
import numpy as np


def google_img_downloader(name):
    filters = dict(
        type='face',
        size='large'
    )

    crawler = GoogleImageCrawler(storage={"root_dir": f"./dataset/{name}"})
    crawler.crawl(
        keyword=name,
        max_num=10,
        filters=filters
    )

    return "Download is complete!"


def train_model_by_img(name):

    # if not os.path.exists(f"dataset/{name}"):
    #     print("[ERROR] there is no directory 'dataset' ")
    #     sys.exit()
    #
    # known_encodings = []
    # images = os.listdir(f"dataset/{name}")

    if not os.path.exists(f"dataset_in_video/{name}"):
        print("[ERROR] there is no directory 'dataset_in_video' ")
        sys.exit()

    known_encodings = []
    images = os.listdir(f"dataset_in_video/{name}")

    # print(images)

    for (i, image) in enumerate(images):
        print(f"[+] processing img {i+1}/{len(images)} ")
        # print(image)

        face_img = face_recognition.load_image_file(f"dataset_in_video/{name}/{image}")
        face_enc = face_recognition.face_encodings(face_img)[0]

        # print(face_enc)

        if len(known_encodings) == 0:
            known_encodings.append(face_enc)
        else:
            for item in range(0, len(known_encodings)):
                result = face_recognition.compare_faces([face_enc], known_encodings[item])
                # print(result)

                if result[0]:
                    known_encodings.append(face_enc)
                    # print("Same person!")
                    break
                else:
                    # print("Other person!")
                    break

    # print(known_encodings)
    # print(f"Length {len(known_encodings)}")

    data = {
        "name": name,
        "encodings": known_encodings
    }

    with open(f"{name}_encodings.pickle", "wb") as file:
        file.write(pickle.dumps(data))

    return f"[INFO] File {name}_encodings.pickle successfully created"


def take_screenshot_from_video(name):

    cap = cv2.VideoCapture("man.mp4")

    count = 0

    if not os.path.exists(f"dataset_in_video/{name}"):
        os.mkdir(f"dataset_in_video/{name}")

    while True:
        ret, frame = cap.read()
        fps = cap.get(cv2.CAP_PROP_FPS)
        multiplier = fps * 3
        print(fps)

        if ret:
            frame_id = int(round(cap.get(1)))
            print(frame_id)
            cv2.imshow("frame", frame)
            k = cv2.waitKey(20)

            if frame_id % multiplier == 0:
                cv2.imwrite(f"dataset_in_video/{name}/{count}_screenshot.jpg", frame)
                print(f"Take a screenshot {count}")
                count += 1
            if k == ord(' '):
                cv2.imwrite(f"dataset_in_video/{name}/{count}_ext_screen.jpg", frame)
                print(f"Take a extra screenshot {count}")
                count += 1
            elif k == ord('q'):
                print("Quit")
                break
        else:
            print("[Error] Can't get the frame...")
            break

    cap.release()
    cv2.destroyAllWindows()


def detect_person_in_video():
    data = pickle.loads(open("man_encodings.pickle", "rb").read())
    # print(data)
    video = cv2.VideoCapture("man.mp4")

    while True:
        ret, image = video.read()

        locations = face_recognition.face_locations(image, model="hog")
        encodings = face_recognition.face_encodings(image, locations)

        for face_encoding, face_location in zip(encodings, locations):
            result = face_recognition.compare_faces(data['encodings'], face_encoding)
            match = None

            if True in result:
                match = data['name']
                print(f"Match found! {match}")

            else:
                print("ACHTUNG!")

            left_top = (face_location[3], face_location[0])
            right_bottom = (face_location[1], face_location[2])
            color = [0, 255, 0]
            cv2.rectangle(image, left_top, right_bottom, color, 4)

            left_bottom = (face_location[3], face_location[2])
            right_bottom = (face_location[1], face_location[2] + 20)
            cv2.rectangle(image, left_bottom, right_bottom, color, cv2.FILLED)
            cv2.putText(
                image,
                match,
                (face_location[3] + 10, face_location[2] + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                4
            )

        cv2.imshow("detect_person_in_video is running", image)
        k = cv2.waitKey(1)

        if k == ord("q"):
            print("quit")
            break


def main():

    # name = "johnny depp"
    # print(google_img_downloader(name=name))
    name = "man"
    # print(train_model_by_img(name=name))
    # take_screenshot_from_video(name)
    detect_person_in_video()


if __name__ == "__main__":
    main()