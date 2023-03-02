# This is a sample Python script.
import os

import openai

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
# from example_generation import *
# from events_precise_generation import *
# from events_triggers_generation_by_GPT3 import *
openai.api_key = os.getenv("OPENAI_API_KEY")

openai.Engine.list()
# from sentence_generation import *
