from flask import Flask
import numpy as np
import document_verifier as dv
# import location_grouper as lg

app = Flask(__name__)


@app.route('/')
def index():
    return "Hello"

@app.route('/verify-document')
def verifyDocument():
    documentType = "ID"
    imagePath = 'https://res.cloudinary.com/deutptnkg/image/upload/v1630854624/abren/uploads/id_cards/1630854623368.jpg'
    profileImagePath = 'https://res.cloudinary.com/deutptnkg/image/upload/v1630854626/abren/uploads/profiles/1630854625900.jpg'
    # id7Pos = np.array([0.06690371, 0.42507288, 0.8374973 , 0.40478554, 0.8485388 ,
    #     0.70656776, 0.07716736, 0.6814791 ])


    data = dv.processDocument(documentType, imagePath, profileImagePath)

    return data

# @app.route('/nearest-rides')
# def nearestRides():
#     data = lg.getNeighbors()

#     return str(data)


if __name__ == '__main__':
    app.run()

