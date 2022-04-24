import algo
import ligne
import station


class Plan:
    def __init__(self):
        self.budget = 0
        self.listStations = []
        self.listLines = []

    def createStation(self, name, x, y):
        self.listStations.append(station.Station(name, x, y))

    def createLine(self, name):
        self.listLines.append(ligne.Lines(name))

    def addStationLine(self, listStationLine, ligneName):
        verif = False
        for i in self.listLines:
            if i.name == ligneName:
                listopti = algo.list_coordinates_and_names(listStationLine)
                i.listStation = listopti
                verif = True
                break
        if not verif:
            print("Ligne non existante")
