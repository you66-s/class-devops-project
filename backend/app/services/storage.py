from sqlalchemy import create_engine, Column, Integer, String, Float, TIMESTAMP
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

Base = declarative_base()

class Prediction(Base):
    __tablename__ = 'predictions'
    id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String(255), nullable=False)
    label = Column(Integer, nullable=False)
    prediction_confiance = Column(Float, nullable=False)
    created_at = Column(TIMESTAMP, default=datetime.utcnow)

class Storage:
    def __init__(self):
        self.__username = "root"
        self.__password = ""
        self.__host = "localhost"
        self.__port = 3306
        self.__database = "emailsdataset"
        self.__url = f"mysql+mysqlconnector://{self.__username}:{self.__password}@{self.__host}:{self.__port}/{self.__database}"
        self.engine = create_engine(self.__url, echo=False)
        Base.metadata.create_all(self.engine)  # créer la table si elle n'existe pas
        self.Session = sessionmaker(bind=self.engine)

    def store_prediction(self, email: str, label: int, prediction_confiance: float):
        session = self.Session()
        try:
            pred = Prediction(email=email, label=label, prediction_confiance=prediction_confiance)
            session.add(pred)
            session.commit()
            print(f"Prediction stored: {email}, label={label}, confiance={prediction_confiance}")
        except Exception as e:
            session.rollback()
            print("Erreur lors du stockage:", e)
        finally:
            session.close()