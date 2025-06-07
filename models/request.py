from pydantic import BaseModel

class Rating(BaseModel):
    userId: str
    targetId: str
    ratingValue: int

class RequestBody(BaseModel):
    userId: str
    ratings: list[Rating]
