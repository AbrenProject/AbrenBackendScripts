import cv2
import face_recognition
import pytesseract
import numpy as np
import re
import urllib.request
# import tflite_runtime.interpreter as tflite

def erode(image):
    kernel = np.ones((3,3),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

def prepareImage(image, pos, scale):
    (h, w) = image.shape[:2]
    t = min((pos[1::2] * h).astype(int)) - 10 * scale
    b = max((pos[1::2] * h).astype(int)) + 10 * scale
    l = min((pos[0::2] * w).astype(int)) - 10 * scale
    r = max((pos[0::2] * w).astype(int)) + 10 * scale
    
    if (b - t) < 100 or (r - l) < 100 or t < 0 or b > h or l < 0 or r > w:
        return None

    return image[t:b, l:r]


def getIDText(result1, result2):
    idCardData = {
        'isValid': True
    }
    
    x = re.search("Full.*Name(.*)\n", result1, re.IGNORECASE)

    if (not x):
        x = re.search("Full.*Name(.*)\n", result2, re.IGNORECASE)
    
    if(not x):
        idCardData['isValid'] = False
        return idCardData

    idCardData['name'] = x.group(1).strip()

    x = re.search("Sex(.*)(.)\n", result2, re.IGNORECASE)
    if (not x):
        x = re.search("Sex(.*)(.)\n", result2, re.IGNORECASE)
        
    if(not x):
        idCardData['isValid'] = False
        return idCardData

    idCardData['sex'] = x.group(2).strip()

    x = re.search("DOB(.*)/(.*\d{4})", result1, re.IGNORECASE)

    if (not x):
        x = re.search("DOB(.*)/(.*\d{4})", result2, re.IGNORECASE)
    
    if(not x):
        idCardData['isValid'] = False
        return idCardData

    idCardData['dateOfBirth'] = x.group(2).strip()


    x = re.search("Issue Dt(.*)/(.*\d{4})", result1, re.IGNORECASE)

    if (not x):
        x = re.search("Issue Dt(.*)/(.*\d{4})", result2, re.IGNORECASE)
    
    if(not x):
        idCardData['isValid'] = False
        return idCardData

    idCardData['issueDate'] = x.group(2).strip()


    x = re.search("Expiry Dt(.*)/(.*\d{4})", result1, re.IGNORECASE)

    if (not x):
        x = re.search("Expiry Dt(.*)/(.*\d{4})", result2, re.IGNORECASE)
    
    if(not x):
        idCardData['isValid'] = False
        return idCardData

    idCardData['expiryDate'] = x.group(2).strip()


    return idCardData

def getDLText(result1, result2):
    dlData = {
        'isValid': True
    }
    
    return dlData

def extractText(documentType, image):
    (h, w) = image.shape[:2]
    cropped1 = image[0 : int(h / 2), 0:w]
    cropped2 = image[int(h / 2) - 20 : h, 0:w]

    eroded1 = erode(cropped1)
    eroded2 = erode(cropped2)

    custom_config = r'-c tessedit_char_whitelist= 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz/  --oem 3 --psm 6'

    result1 = pytesseract.image_to_string(eroded1, config=custom_config)


    result2 = pytesseract.image_to_string(eroded2, config=custom_config)
#    print(result1)

#     print("----------------------")
#     print(result2)
    
#     print("----------------------")
#     print("----------------------")
    
#     cv2.imshow('eroded1', eroded1)
#     cv2.waitKey(0)
#     cv2.imshow('eroded2', eroded2)
#     cv2.waitKey(0)

    return getIDText(result1, result2) if documentType == "ID" else getDLText(result1, result2)

def verifyFace(documentType, image, profileImage):
    imageEncodings = face_recognition.face_encodings(image)
    profileEncodings = face_recognition.face_encodings(profileImage)
    
    if(documentType == "ID" and len(imageEncodings) < 1):
        return False

    if(len(profileEncodings) != 1):
        return False    
    
    matches = face_recognition.compare_faces(imageEncodings, profileEncodings[0])
    
    return all(matches)

def processDocument(documentType, imagePath, profileImagePath, pos):
    # image = io.imread(imagePath)
    # profileImage = io.imread(profileImagePath)
    
    resp = urllib.request.urlopen(imagePath)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    resp2 = urllib.request.urlopen(profileImagePath)
    profileImage = np.asarray(bytearray(resp2.read()), dtype="uint8")
    profileImage = cv2.imdecode(profileImage, cv2.IMREAD_COLOR)
    
    data = {
        'isVerified' : False,
        'isLogoVerified': False,    
        'isTextVerified': False,
        'isFaceVerified': False,
    }
    
    scale = 0
    while(not data['isTextVerified']):
        idImage = prepareImage(image, pos, scale)
        
        if(idImage is None):
            break
            
        textData = extractText(documentType, idImage)
        if documentType == "ID":
            data['idCardData'] = textData
        else:
            data['drivingLicenseData'] = textData
        
        data['isTextVerified'] = textData['isValid']
        
        scale += 1
        
    scale = 0
    while(not data['isTextVerified']):
        idImage = prepareImage(image, pos, scale)
        
        if(idImage is None):
            break
        textData = extractText(documentType, idImage)
        if documentType == "ID":
            data['idCardData'] = textData
        else:
            data['drivingLicenseData'] = textData
        
        data['isTextVerified'] = textData['isValid']
        
        scale -= 1
        
    if(data['isTextVerified']):
        data['isFaceVerified'] = verifyFace(documentType, image, profileImage)
        
    return data
