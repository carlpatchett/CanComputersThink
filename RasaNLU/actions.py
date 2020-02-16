# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/core/actions/#custom-actions/


# This is a simple example for a custom action which utters "Hello World!"

# from typing import Any, Text, Dict, List
#
# from rasa_sdk import Action, Tracker
# from rasa_sdk.executor import CollectingDispatcher
#
#
# class ActionHelloWorld(Action):
#
#     def name(self) -> Text:
#         return "action_hello_world"
#
#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
#
#         dispatcher.utter_message(text="Hello World!")
#
#         return []

import rasa_sdk
import spacy
from spacy import displacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
from rasa_sdk import Action
from rasa_sdk.events import SlotSet


class RespondName(Action):
    """Stores the bot use case in a slot"""

    nlp = spacy.load('en_core_web_sm')

    def name(self):
        return "action_respond_name"

    def run(self, dispatcher, tracker, domain):

        # we grab the whole user utterance here as there are no real entities
        # in the use case
        message = tracker.latest_message.get('text')
        nlp = spacy.load('en_core_web_sm')
        doc = nlp(message)

        docText = None
        for ent in doc.ents:
            docText = ent.text
        
        return [SlotSet('name', docText)]