
from fastapi import (FastAPI, 
                    Depends, 
                    HTTPException, 
                    status, 
                    Request, 
                    File, 
                    UploadFile)
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from starlette.routing import Host
from pydantic import BaseModel
from tortoise import fields
from tortoise.contrib.fastapi import register_tortoise
from tortoise.models import Model
from tortoise.contrib.pydantic import pydantic_model_creator
from passlib.hash import bcrypt
from PIL import Image
from fmnist.fmnist import predict_from_image
import uvicorn
import pickle
import jwt
import os, io

app = FastAPI()
JWT_SECRET = 'jwtsecret'
oauth2_scheme = OAuth2PasswordBearer(tokenUrl='token')
PATH = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(PATH, f"train/models/20211015_005306.pkl")

# model = FMNIST()

picklefile = open(model_path,"rb")
classifier=pickle.load(picklefile)


# 2. Class which describes Bank Notes measurements
class BankNote(BaseModel):
    variance: float 
    skewness: float 
    curtosis: float 
    entropy: float
    
class User(Model):
    id = fields.IntField(pk=True)
    username = fields.CharField(50, unique=True)
    password_hash = fields.CharField(128)

    @classmethod
    def get_user(self, cls, username):
        return cls.get(username=username)

    # @classmethod
    def verify_password(self, password):
        return bcrypt.verify(password, self.password_hash)

User_Pydantic = pydantic_model_creator(User, name='User')
UserIn_Pydantic = pydantic_model_creator(User, name='UserIn', exclude_readonly=True)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl='token')

async def authenticate_user(username: str, password: str):
    user = await User.get(username=username)
    if not user:
        return False
    if not user.verify_password(password):
        return False
    return user

@app.post('/token')
async def generate_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = await authenticate_user(form_data.username, form_data.password)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, 
            detail='Invalid Username or Password'
            )
    user_obj = await User_Pydantic.from_tortoise_orm(user)
    token = jwt.encode(user_obj.dict(), JWT_SECRET)
    return {'access_token': token, 'token_type': 'bearer'}

@app.post('/users', response_model=User_Pydantic)
async def create_user(user: UserIn_Pydantic):
    user_obj = User(username=user.username, password_hash=bcrypt.hash(user.password_hash))
    await user_obj.save()
    return await User_Pydantic.from_tortoise_orm(user_obj)

async def get_user_current(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
        user = await User.get(id=payload.get('id'))
    except:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, 
            detail='Invalid Username or Password'
            )
    return await User_Pydantic.from_tortoise_orm(user)

@app.get('/users/me', response_model=User_Pydantic)
async def get_user(user: User_Pydantic = Depends(get_user_current)):
    return user

@app.post('/predict', response_model=str)
async def predict_banknote(data: BankNote,
         user: User_Pydantic = Depends(get_user_current)):
    ''' Machine Learning model '''
    variance = data.variance
    skewness = data.skewness
    curtosis = data.curtosis
    entropy  = data.entropy
    prediction = classifier.predict([[variance,skewness,curtosis,entropy]])
    if(prediction[0]>0.5):
        status="Fake bank note"
    else:
        status="Correct bank note"
    return status

@app.post("/predicttest")
def predict(request: Request, file:UploadFile=File(...)):
    content = file.file.read()
    image = Image.open(io.BytesIO(content)).convert('L')
    output = predict_from_image(image)
    return output

register_tortoise(
    app, 
    db_url='sqlite://db.sqlite3',
    modules={'models': ['main']},
    generate_schemas=True,
    add_exception_handlers=True
    )

if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=8000)
# uvicorn main:app --reload