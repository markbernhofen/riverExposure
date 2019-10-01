#####################################################################
#   floodMap.py
#
#   This code produces a flood map using a methodology which is similar to the Height Above Nearest Drainage (HAND)
#   method developed by Nobre et. al (2011). This script produces three final flood maps. 1. A HAND Map 2. A map of the
#   upstream accumulating area of the river that caused the flooding 3. A map of the Strahler stream order of the river
#   that caused the flooding.
#
#   This method differs from the traditional HAND methodology in that it splits the analysis into the different Strahler
#   stream orders that exist within the river network. Each Strahler stream order can have a different "HAND" height, as
#   long as the "HAND" height of a smaller order doesn't exceed that of a larger order. The analysis is run for each
#   Strahler stream order in turn and the final results are merged into the final output files. The influence of higher
#   order streams takes precedence over smaller order streams. This is important at river confluences and for
#   multi-channel rivers, where the largest (main) channel has the greatest effect on flooding. As similar order streams
#   may have similar flooding potential, the influence of a larger order stream over a smaller order stream will only
#   take effect if there is a difference of at least 2 Strahler stream orders between them
#
#   INPUTS
#   built to use the following MERIT (Yamazaki) datasets:
#   + Hydrologically Adjusted Digital Elevation Model (save as dem.tif)
#   + Flow direction (save as dir.tif)
#   + Upstream drainage area (save as acc.tif)
#   ^ available from: http://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_Hydro/
#   this file should have been created using the strahler.py code:
#   + Strahler file (save as strahler.tif)
#   this file can be created in GIS:
#   + a simplified version of the koeppen-geiger climate zone dataset (save as climate.tif)
#   > zones simplified as follows [new]=(old): [1]=(1-3), [2]=(4-5), [3]=(6-7), [4]=(8-16), [5]=(17-28), [6]=(29-30)
#   ^ original data available from: http://koeppen-geiger.vu-wien.ac.at/present.htm
#
#   NOTE: Directory must contain same size / resolution / location files in notation.tif specified above
#
#   OUTPUTS
#   this code will output the following rasters:
#   + Height Above Nearest Drainage flood map (saved as hand.tif)
#   + Upstream accumulating area of river causing flooding (saved as floodAcc.tif)
#   + Strahler order of river causing flooding (saved as floodStrahler.tif
#
#   Requirements: GDAL 1.8+, numpy
#
#   This script is written to be run from the command line.
#   Before running the script, make sure you check in __init__ that the HAND heights for each Strahler order for the
#   different climate zones are as you want them.
#   You need to specify your chosen stream initiation upstream drainage area threshold (in km^2), the default is 5 km^2
#
#   Ensure the terminal is in the directory that contains the floodMap.py script. Then run as below where (-m) is the
#   chosen stream initiation drainage threshold and (-o) is overwrite (optional)
#
#   python floodMap.py -i C:\path\to\directory\with\files -m 5 -o
#
#   Author: Mark Bernhofen (2019) cn13mvb@leeds.ac.uk
#####################################################################

import argparse
import os
import sys
from datetime import datetime

import numpy as np
from osgeo import gdal



class floodMap:

    def __init__(self, inputDir, overwrite, accThresh):

        # Here, set the Strahler order HAND heights for the different climate zones. Climate zone labels:
        # 1 = Tropical, 2 = Arid, 3 = Semi-Arid, 4 = Temperate, 5 = Cold, 6 = Polar
        self.orders = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.climateHeights = {1: [1, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12],
                               2: [1, 1, 1, 1, 2, 2, 3, 4, 7, 11, 11],
                               3: [1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 10],
                               4: [1, 1, 1, 3, 4, 5, 6, 7, 7, 8, 10],
                               5: [1, 1, 1, 3, 4, 5, 6, 7, 7, 8, 10],
                               6: [1, 1, 1, 3, 4, 5, 6, 7, 7, 9, 10]}

        self.inputDir = inputDir
        if not os.path.exists((self.inputDir)):
            print("Input directory doesn't exist")
            sys.exit(-1)
        self.outputDir = os.path.join(inputDir, "Outputs")
        if not os.path.exists(self.outputDir):
            os.makedirs(self.outputDir)
        self.overwrite = overwrite
        self.accThresh = int(accThresh)

        # For neighbourhood analysis we want to find which of the surrounding cells
        # drains into the center cell. Therefore need to invert drainage direction
        # 8-flow. Starting by looking East.
        self.drainInToDirectionValues = [16, 32, 64, 128, 1, 2, 4, 8]   # cells draining IN TO cell
        self.pixelCounter = 0
        self.riverCellCounter = 0

        # Define input filenames and then read in the data.
        self.dirFile = os.path.join(self.inputDir, 'dir.tif')
        self.dir = self.readData(self.dirFile).astype(np.uint8)
        self.demFile = os.path.join(self.inputDir, 'dem.tif')
        self.dem = self.readData(self.demFile, saveInfo=True)
        self.accFile = os.path.join(self.inputDir, 'acc.tif')
        self.acc = self.readAccData(self.accFile)
        self.strahlerFile = os.path.join(self.inputDir, 'strahler.tif')
        self.strahler = self.readData(self.strahlerFile)
        self.climateFile = os.path.join(self.inputDir, 'climate.tif')
        self.climate = self.readData(self.climateFile).astype(np.uint8)
        self.watMask = self.readAccData(self.accFile).astype(np.bool)
        self.handFile = os.path.join(self.resultsDir, "hand.tif")
        self.floodAccFile = os.path.join(self.resultsDir, "floodAcc.tif")
        self.floodStrahlerFile = os.path.join(self.resultsDir, "floodStrahler.tif")

    def readData(self, fileName, saveInfo=False):
        fileHandle = gdal.Open(fileName)
        if fileHandle is None:
            print('ERROR: data file no data: ', fileName)
            sys.exit(-1)

        self.rasterXSize = fileHandle.RasterXSize
        self.rasterYSize = fileHandle.RasterYSize

        if saveInfo:
            self.dataInfo = fileHandle

        band = fileHandle.GetRasterBand(1)
        data = band.ReadAsArray(0, 0, fileHandle.RasterXSize, fileHandle.RasterYSize)

        fileHandle = None
        return data

    def readAccData(self, fileName):
        fileHandle = gdal.Open(fileName)
        if fileHandle is None:
            print('ERROR: data file no data: ', fileName)
            sys.exit(-1)

        accArray = np.array(fileHandle.GetRasterBand(1).ReadAsArray())
        # Applying the drainage threshold
        data = np.where(accArray>self.accThresh, accArray, 0)

        fileHandle = None
        return data

    def mainProcess(self):

        # Create an empty hand array that will be filled throughout the analysis
        self.hand = np.full(self.dir.shape, 0, dtype=np.int16)

        # Create an empty floodAcc array that will be filled throughout the analysis
        self.floodAcc = np.full(self.dir.shape, 0, dtype=np.int32)

        # Create an empty floodStrahler array that will be filled throughout the analysis
        self.floodStrahler = np.full(self.dir.shape, 0, dtype=np.int16)

        # How many pixels are there in the water mask?
        numWaterPix = np.sum(self.watMask)

        # Create an array numWaterPix long with values for row, col, height, drainage, order, and climate
        wph = np.empty((numWaterPix), dtype=[('row', int), ('col', int), ('height', int), ('drainage', int),
                                             ('order', int), ('climate', int)])

        # Here we are going to fill the wph array with the corresponding values for every cell in our river network
        for r in range(self.rasterYSize):
            for c in range(self.rasterXSize):
                if self.watMask[r, c]:      # if it exists (imported as boolean earlier)
                    elev = self.dem[r, c]
                    acc = self.acc[r, c]
                    order = self.strahler[r, c]
                    climate = self.climate[r, c]
                    if order == 0:
                        continue    # In some coastal areas the acc stream is one cell longer than the Strahler stream
                    self.floodAcc[r, c] = acc  # Burn river acc cells into floodAcc array
                    self.hand[r, c] = 1     # river values in HAND will be 1
                    self.floodStrahler[r, c] = order
                    # fill the wph array with the relevant values
                    wph[self.riverCellCounter] = (r, c, elev, acc, order, climate)
                    self.riverCellCounter += 1


        # Going to work through the river network one Strahler order at a time, beginning at the maximum and finishing
        # at 1.
        maxStrahler = int(np.amax(self.strahler))
        for i in range (maxStrahler, 0, -1):
            currentOrder = i
            wpc = wph[wph['order']==currentOrder]
            # Sort the river cells based on height
            wpc.sort(order='height')
            length = len(wpc)

            for i in range(length):
                # start at highest elevation
                wpx = wpc[length - i - 1]
                r = wpx['row']
                c = wpx['col']
                ht = wpx['height']
                dr = wpx['drainage']
                so = wpx['order']
                cl = wpx['climate']

                dir = self.dir[r, c]
                if dir == 247:      # undefined
                    self.hand[r, c] = 255
                if dir == 255:      # inland depression
                    self.hand[r, c] = 255
                else:
                    self.pixelCounter += 1
                    self.riverCellCounter += 1

                    # Here going to set the "hand height" based on the climate zone and Strahler order of the river
                    handHeight = self.chooseHandHeight(cl, r, c)

                    neighbors = self.neighborhood(r, c)
                    for idx, n in enumerate(neighbors):
                        if n is not None:
                            pixel = (n[0], n[1], idx)
                            self.neighborProcess(pixel, ht, dr, so, handHeight)

    def neighborhood(self, row, col):
        '''
        This function returns a vector containing the coordinates of the current pixel's 8
        neighbors. The neighboring elements are labelled as follows:
        5  6  7
        4  x  0
        3  2  1
        this matches the flow directions defined in the previous __init__ function
        '''
        neighbors = [[row, col + 1], [row + 1, col + 1], [row + 1, col], [row + 1, col - 1], [row, col - 1],
                     [row - 1, col - 1], [row - 1, col], [row - 1, col + 1]]
        for i in range(8):
            neighborR, neighborC = neighbors[i]
            if(neighborR < 0) or (neighborR > self.rasterYSize - 1):
                neighbors[i] = None
            elif (neighborC < 0) or (neighborC > self.rasterXSize - 1):
                neighbors[i] = None

        return neighbors

    def neighborProcess(self, neighbor, riverHeight, riverAcc, riverOrder, handHeight):
        '''
        This function processes the current pixel and checks the neighboring pixels for the one that drains
        into it. It then works out the height above the draining river cell and inserts it into hand array.
        Also copies river accumulation value and Strahler stream order onto all upstream draining cells (within the
        HAND height limit).
        '''
        q = [neighbor]
        while len(q) > 0:
            neighbor = q.pop(0)
            row = neighbor[0]
            col = neighbor[1]
            index = neighbor[2]
            maxHeight = handHeight
            order = riverOrder
            hand = self.hand[row, col]
            dir = self.dir[row, col]

            if dir == 247:      # undefined
                self.hand[row, col] = 255
                continue
            if dir == 255:      # inland depression
                self.hand[row, col] = 255
                continue

            # Don't want to overwrite our layers if they are the same stream order.
            if hand > 0 and self.floodStrahler[row, col] == order:
                continue

            # Does the neighboring cell drain into the current one?
            if dir == self.drainInToDirectionValues[index]:

                # Going to keep current river's order and upstream accumulating area if existing stream is only one
                # Strahler order larger.
                if hand > 0 and self.floodStrahler[row, col] == order + 1:
                    # Keep hand as 1 if it is already 1
                    if hand == 1:
                        self.floodStrahler[row, col] = order
                        if riverAcc < 0:
                            print('DEBUG: riverAcc is a negative value')
                            sys.exit(-1)
                        self.floodAcc[row, col] = riverAcc
                    else:
                        neighborHeight = self.dem[row, col]
                        relativeHeight = int(neighborHeight - riverHeight)
                        # if relative height is smaller or equal to existing hand height (+1 because hand heights are
                        # all one value larger due to river cells having a value of 1)
                        if (relativeHeight + 1) <= hand:
                            self.hand[row, col] = relativeHeight + 1
                            self.extrStrahler[row, col] = order
                            if riverAcc < 0:
                                print('DEBUG: riverAcc is a negative value')
                                sys.exit(-1)
                            self.extrAcc[row, col] = riverAcc

                    neighbors = self.neighborhood(row, col)
                    for index, n in enumerate(neighbors):
                        if n is not None:
                            pixel = (n[0], n[1], index)
                            q.append(pixel)

                # Not going to overlap floodAcc and floodStrahler information if the existing stream is more than one
                # Strahler stream order than the current stream. However, the hand heights will be adjusted to account
                # for the existing (smaller stream).
                elif hand > 0 and self.floodStrahler[row, col] > order+1:
                    # Keep hand as 1 if it is already 1
                    neighborHeight = self.dem[row, col]
                    relativeHeight = int(neighborHeight - riverHeight)
                    if hand > 1 and (relativeHeight + 1) <= hand:
                        self.hand[row, col] = relativeHeight + 1

                    neighbors = self.neighborhood(row, col)
                    for index, n in enumerate(neighbors):
                        if n is not None:
                            pixel = (n[0], n[1], index)
                            q.append(pixel)

                # Keep hand as 1 if it is currently one
                elif hand == 1:
                    # Debug
                    self.extrStrahler[row, col] = order
                    self.extrAcc[row, col] = riverAcc
                    neighbors = self.neighborhood(row, col)
                    for index, n in enumerate(neighbors):
                        if n is not None:
                            pixel = (n[0], n[1], index)
                            q.append(pixel)

                # If the neighboring draining cell hasn't already been processed
                else:
                    neighborHeight = self.dem[row, col]
                    relativeHeight = int(neighborHeight - riverHeight)
                    if relativeHeight <= maxHeight:
                        self.hand[row, col] = relativeHeight + 1
                        self.floodStrahler[row, col] = order
                        # Debug
                        if riverAcc < 0:
                            print('DEBUG: riverAcc is a negative value')
                            sys.exit(-1)
                        self.floodAcc[row, col] = riverAcc
                        self.pixelCounter += 1

                        neighbors = self.neighborhood(row, col)
                        for index, n in enumerate(neighbors):
                            if n is not None:
                                pixel = (n[0], n[1], index)
                                q.append(pixel)

    def chooseHandHeight(self, climate, r, c):
        '''
        Return the HAND height for the given river cell based on its Strahler order and climate zone
        '''

        # If climate dataset shows 0 (which might be the case if there is not perfect overlap). The climate will be
        # calculated based on the surrounding climate
        if climate == 0:
            neighboringClimate = self.neighboringClimate(r, c)
            handHeight = self.climateHeights[neighboringClimate]
            return handHeight
        else:
            handHeight = self.climateHeights[climate]
            return handHeight

    def neighboringClimate(self, row, column):
        '''
        If the climate dataset shows 0. The current climate zone will be calculated based on the surrounding cells. The
        surrounding neighborhoods (up to 1000 neighborhoods) will be searched for climate data. If none is found, then
        the most common (mode) climate value for the current row will be used (unlikely to need this step).
        '''

        neighbors = self.neighborhood(row, column)
        values = []
        for n in neighbors:
            if n is not None:
                r = n[0]
                c = n[1]
                climateVal = self.climate[r, c]
                if isinstance(climateVal, int):
                    values.append(int(climateVal))
                else:
                    continue

        # If values list is empty look to a larger neighborhood
        neighborhoodNum = 2
        while not values:

            neighbors = self.multiNeighborhood(row, column, neighborhoodNum)
            for n in neighbors:
                if n is not None:
                    # Debug
                    r = n[0]
                    c = n[1]
                    climateVal = self.climate[r, c]
                    if climateVal > 0:
                        values.append(int(climateVal))
                    else:
                        continue
            neighborhoodNum += 1

            # To avoid long run-times limit the number of neighborhoods to analyze to 1000
            if neighborhoodNum == 1000:
                break

        if values:
            neighborClimate = max(set(values), key=values.count)
            self.climate[row, column] = neighborClimate
            return neighborClimate

        # If multi neighborhood method hasn't worked then just calculate the mode for the current row of the array
        # (and further rows if the current row only has zero values (unlikely))
        if not values:
            neighborClimate = False
            rowNum = 0
            while neighborClimate is False:
                currentRow = row - rowNum
                if currentRow > 0:
                    isoRow = self.climate[currentRow, :]
                    isoRow = isoRow[isoRow != 0]
                    isoRow = list(isoRow)
                    if isoRow:
                        neighborClimate = max(set(isoRow), key=isoRow.count)
                        self.climate[row, column] = neighborClimate
                        return neighborClimate
                currentRow = row + rowNum
                if currentRow < self.rasterYSize:
                    isoRow = self.climate[currentRow, :]
                    isoRow = isoRow[isoRow != 0]
                    isoRow = list(isoRow)
                    if isoRow:
                        neighborClimate = max(set(isoRow), key=isoRow.count)
                        self.climate[row, column] = neighborClimate
                        return neighborClimate
                rowNum += 1

    def multiNeighborhood(self, row, col, m):
        '''
        Function for the neighboringClimate function. Finds cells in larger neighborhoods surrounding cell in question.
        Rather than the 8 cell neighborhood in the traditional neighborhood function above.
        '''
        neighbors = []

        for i in range(m + 1):
            neighbor1 = [row + m, col + i]
            neighbors.append(neighbor1)
            neighbor2 = [row - m, col + i]
            neighbors.append(neighbor2)
            neighbor3 = [row + m, col - i]
            neighbors.append(neighbor3)
            neighbor4 = [row - m, col - i]
            neighbors.append(neighbor4)
            neighbor5 = [row + i, col + m]
            neighbors.append(neighbor5)
            neighbor6 = [row - i, col + m]
            neighbors.append(neighbor6)
            neighbor7 = [row + i, col - m]
            neighbors.append(neighbor7)
            neighbor8 = [row - i, col - m]
            neighbors.append(neighbor8)

        for i in range(len(neighbors)):
            neighborR, neighborC = neighbors[i]
            if (neighborR < 0) or (neighborR > self.rasterYSize - 1):
                neighbors[i] = None
            elif (neighborC < 0) or (neighborC > self.rasterXSize - 1):
                neighbors[i] = None

        return neighbors

    def writeTifOutput(self):
        print(str(datetime.now()), 'Saving GeoTiffs...')
        print("river cells= ", self.riverCellCounter)
        print("processed cells= ", self.pixelCounter)
        print("total pixels= ", self.rasterXSize * self.rasterYSize)
        print("HAND pixels= ", np.count_nonzero(self.hand))

        # Write text file with layer information
        fLoc = os.path.join(self.resultsDir, "read_me.txt")
        f = open(fLoc, "w+")
        f.write("Stream initiation upstream drainage area threshold: " + str(self.accThresh) + "\n")
        f.write("Strahler orders " + str(self.orders) + "\n")
        f.write("Tropical (1) HAND heights " + str(self.HHeights1) + "\n")
        f.write("Arid (2) HAND heights" + str(self.HHeights2) + "\n")
        f.write("Semi-Arid (3) HAND heights" + str(self.HHeights3) + "\n")
        f.write("Temperate (4) HAND heights" + str(self.HHeights4) + "\n")
        f.write("Cold (5) HAND heights" + str(self.HHeights5) + "\n")
        f.write("Polar (6) HAND heights" + str(self.HHEights6) + "\n")
        f.close()

        driver = gdal.GetDriverByName("GTiff")
        dstHand = driver.Create(self.handFile, self.rasterXSize, self.rasterYSize, 1, gdal.GDT_Int16)
        dstFloodAcc = driver.Create(self.floodAccFile, self.rasterXSize, self.rasterYSize, 1, gdal.GDT_Int32)
        dstFloodStrahler = driver.Create(self.floodStrahlerFile, self.rasterXSize, self.rasterYSize, 1, gdal.GDT_Int16)
        bandH = dstHand.GetRasterBand(1)
        bandH.WriteArray(self.hand, 0, 0)
        bandA = dstFloodAcc.GetRasterBand(1)
        bandA.WriteArray(self.floodAcc, 0, 0)
        bandS = dstFloodStrahler.GetRasterBand(1)
        bandS.WriteArray(self.floodStrahler, 0, 0)

        projection = self.dataInfo.GetProjection()
        geotransform = self.dataInfo.GetGeoTransform()

        dstHand.SetGeoTransform(geotransform)
        dstFloodAcc.SetGeoTransform(geotransform)
        dstFloodStrahler.SetGeoTransform(geotransform)
        dstHand.SetProjection(projection)
        dstFloodAcc.SetProjection(projection)
        dstFloodStrahler.SetProjection(projection)
        dstHand.FlushCache()
        dstFloodAcc.FlushCache()
        dstFloodStrahler.FlushCache()

        dstHand = None
        dstFloodAcc = None
        dstFloodStrahler = None
        self.dataInfo = None

        return

#############################################################################
# Main
#
# floodMap.py -i 'input_directory' -m 5

if __name__ == "__main__":
    version_num = int(gdal.VersionInfo('VERSION_NUM'))
    if version_num < 1800: # because of GetGeoTransform(can_return_null)
        print('ERROR: Python bindings of GDAL 1.8.0 or later required')
        sys.exit(-1)

    parser = argparse.ArgumentParser(description='Generate Flood Maps')
    apg_input = parser.add_argument_group('Input')
    apg_input.add_argument("-i", "--inputdir", nargs='?',
                           help="Filepath to directory containing all 5 required input files")
    apg_input.add_argument("-o", "--overwrite", action='store_true',
                           help="Will overwrite any existing files found in hand directory")
    apg_input.add_argument("-m", "--minthreshold", default=5, nargs='?',
                           help="Minimum drainage area threshold to consider [kilometres squared]")
    options = parser.parse_args()

    FM = floodMap(inputDir=options.inputdir, overwrite=options.overwrite, accThresh=options.minthreshold)

    if not os.path.isfile(FM.handFile) or options.overwrite:
        FM.mainProcess()
        FM.writeTifOutput()
    else:
        print('hand.tif exists', H.handFile)

    if options.verbose:
        print(str(datetime.now()), "Done.")

################################################################################