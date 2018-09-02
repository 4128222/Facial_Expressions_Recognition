import face_recognition

def discern_face(image):
    face_locations = face_recognition.face_locations(image)
    if len(face_locations) == 0:
        return None
    else:
        # 对图片进行预处理, 并将其存储在数据列表中
        # 得到面部坐标
        top = face_locations[0][0]
        right = face_locations[0][1]
        bottom = face_locations[0][2]
        left = face_locations[0][3]
        image = image[top - 20:bottom + 20,left - 20:right + 20]
        return image
