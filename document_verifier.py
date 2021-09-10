import cv2
import face_recognition
import pytesseract
import numpy as np
import re
import urllib.request
import tflite_runtime.interpreter as tflite
from datetime import datetime

def extractDocument(image):
    inputImage = cv2.resize(image, (224, 224))
    inputImage = inputImage[...,::-1].astype(np.float32) / 255.0
    inputImage = np.expand_dims(inputImage, axis=0)
    
    interpreter = tflite.Interpreter(model_path="model_id_dl.tflite")
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], inputImage)

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])

    return output_data[0]


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

    try:
        issueDate = datetime.strptime(idCardData['issueDate'], '%b %d, %Y')
    except ValueError:
        try:
            issueDate = datetime.strptime(idCardData['expiryDate'].split(',')[0] + "," + idCardData['issueDate'].split(', ')[1], '%b %d, %Y')
        except:
            idCardData['isValid'] = False
            return idCardData

    try:
        expiryDate = datetime.strptime(idCardData['issueDate'].split(',')[0] + ", " + idCardData['expiryDate'].split(', ')[1], '%b %d, %Y')       
    except:
        idCardData['isValid'] = False
        return idCardData

    now = datetime.now()
    if issueDate > now or expiryDate < now:
        idCardData['isValid'] = False
        return idCardData

    return idCardData

def getDLText(result1, result2):
    dlData = {
        'isValid': True
    }

    x = re.search("(\d{1,2}/\d{1,2}/\d{4}).*(\d{1,2}/\d{1,2}/\d{4})", result1, re.IGNORECASE)

    if (not x):
        x = re.search("(\d{1,2}/\d{1,2}/\d{4}).*(\d{1,2}/\d{1,2}/\d{4})", result2, re.IGNORECASE)
    
    if(not x):
        dlData['isValid'] = False
        return dlData
    
    dlData['issueDate'] = x.group(1)
    dlData['expiryDate'] = x.group(2)
    
    try:
        issueDate = datetime.strptime(dlData['issueDate'], '%d/%m/%Y')
        expiryDate = datetime.strptime(dlData['expiryDate'], '%d/%m/%Y')
    except ValueError:
        dlData['isValid'] = False
        return dlData
    
    now = datetime.now()
    if now >= datetime(now.year, 1, 1) and now <= datetime(now.year, 9, 10):
        now = datetime(now.year - 8, now.month - 8, now.day)
    else:
        now = datetime(now.year - 7, now.month - 8, now.day)

    if issueDate > now or expiryDate < now:
        dlData['isValid'] = False
        return dlData
    
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

    # print(result1)

    # print("----------------------")
    # print(result2)
    
    # print("----------------------")
    # print("----------------------")

    return getIDText(result1, result2) if documentType == "ID" else getDLText(result1, result2)

def verifyFace(image, profileImage):
    imageEncodings = face_recognition.face_encodings(image)
    profileEncodings = face_recognition.face_encodings(profileImage)
    
    if(len(imageEncodings) < 1):
        return False

    if(len(profileEncodings) != 1):
        return False    
    
    matches = face_recognition.compare_faces(imageEncodings, profileEncodings[0])
    
    return all(matches)

def processDocument(documentType, imagePath, profileImagePath):    
    resp = urllib.request.urlopen(imagePath)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    resp2 = urllib.request.urlopen(profileImagePath)
    profileImage = np.asarray(bytearray(resp2.read()), dtype="uint8")
    profileImage = cv2.imdecode(profileImage, cv2.IMREAD_COLOR)
        
    pos = extractDocument(image)    

    data = {
        'isVerified' : False,
        'isLogoVerified': True, #TODO: Change    
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
        data['isFaceVerified'] = verifyFace(image, profileImage)

    if(data['isTextVerified'] and data['isFaceVerified'] and data['isLogoVerified']):
        data['isVerified'] = True
        
    return data
