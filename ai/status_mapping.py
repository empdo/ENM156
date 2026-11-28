
LABEL_TO_STATUS = {
    "felm4": "fault1",
    "korrm4": "normal",
    "medm8": "normal",
    "utanm8": "fault2",
    "bakgrund": "idle",
    "background": "idle",
}


def interpret_prediction(prediction):
    status = LABEL_TO_STATUS.get(prediction, "uncertain")
    names = {"normal": "OK", "fault1": "FEL M4", "fault2": "FEL M8",
             "idle": "INAKTIV", "uncertain": "OSÄKER"}
    return {"label": prediction, "status": status, "meaning": names[status]}
