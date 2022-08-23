import pickle

import face_recognition
from PIL import Image, ImageDraw
import os


def face_rec(img_path):
    """ find image coordinates """
    """находим координаты изображения"""
    img = face_recognition.load_image_file(img_path)
    img_face_locations = face_recognition.face_locations(img)

    return img, img_face_locations


def draw_img(img, img_face_locations, img_path):
    """ frame the face in the image """
    """ выделяет в рамку лицо на изображении """

    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)

    for (top, right, bottom, left) in img_face_locations:
        draw.rectangle(((left, top), (right, bottom)), outline=(255, 255, 0), width=8)

    del draw

    pil_img.save(f"img/new_{img_path.split('/')[1]}")

    return "Drawing image is complete"


def extracting_faces(img, img_face_locations, img_path):
    """ cut out a face from a photo """
    """ вырезает лицо с фотографии """

    if not os.path.exists(f"img/{img_path.split('/')[1].split('.')[0]}"):
        os.mkdir(f"img/{img_path.split('/')[1].split('.')[0]}")

    count = 0
    for face_location in img_face_locations:
        top, right, bottom, left = face_location

        face_img = img[top:bottom, left:right]
        pil_img = Image.fromarray(face_img)
        pil_img.save(f"img/{img_path.split('/')[1].split('.')[0]}/{count}_face_img.jpg")
        count += 1

    return "Extracting face(s) is complete"


def compare_faces(img_path_1, img_path_2):
    """ compare faces """
    """сравниваем лица"""
    img1 = face_recognition.load_image_file(img_path_1)
    img1_encodings = face_recognition.face_encodings(img1)[0]

    img2 = face_recognition.load_image_file(img_path_2)
    img2_encodings = face_recognition.face_encodings(img2)[0]

    result = face_recognition.compare_faces([img1_encodings], img2_encodings)

    return result



def main():

    # img_path = "img/arnold.jpg"
    # print(face_rec(img_path=img_path))
    # img, img_face_locations = face_rec(img_path=img_path)
    # print(f"Found {len(img_face_locations)} face(s) in this image")
    # print(draw_img(img=img, img_face_locations=img_face_locations, img_path=img_path))
    # print(extracting_faces(img=img, img_face_locations=img_face_locations, img_path=img_path))

    # img_path = "img/face_people.jpg"
    # print(face_rec(img_path=img_path))
    # img, img_face_locations = face_rec(img_path=img_path)
    # print(f"Found {len(img_face_locations)} face(s) in this image")
    # print(draw_img(img=img, img_face_locations=img_face_locations, img_path=img_path))
    # print(extracting_faces(img=img, img_face_locations=img_face_locations, img_path=img_path))

    img_path = "img/people1.jpeg"
    print(face_rec(img_path=img_path))
    img, img_face_locations = face_rec(img_path=img_path)
    print(f"Found {len(img_face_locations)} face(s) in this image")
    print(draw_img(img=img, img_face_locations=img_face_locations, img_path=img_path))
    print(extracting_faces(img=img, img_face_locations=img_face_locations, img_path=img_path))

    # ------------------------------------------------------------------------------------------

    # img_path_1 = "img/arnold.jpg"
    # img_path_2 = "img/arni2.jpeg"

    # img_path_1 = "img/friends1.jpg"
    # img_path_2 = "img/friends2.jpeg"

    # img_path_1 = "img/usyk.jpeg"
    # img_path_2 = "img/usyk2.jpg"
    #
    # print(compare_faces(img_path_1=img_path_1, img_path_2=img_path_2))


if __name__ == "__main__":
    main()