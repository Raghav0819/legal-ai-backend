from db.base import Base
from db.session import engine

from db.models.conversation import Conversation
from db.models.message import Message

print("Creating database tables...")

Base.metadata.create_all(bind=engine)

print("Done.")