#####################################################################
#   strahler.py
#
#   this code assigns a Strahler order to each stream in the river network
#   The strahler.py file that is created by this script is needed to run the
#   subsequent floodMap.py script
#
#   INPUTS
#   built to use the following MERIT Hydro datasets:
#   + Flow direction (save as dir.tif)
#   + Upstream drainage area (save as acc.tif)
#   available from: http://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_Hydro/
#
#   NOTE: Directory must contain same size / resolution / location files in notation.tif specified above
#
#   OUTPUTS
#   this code will output the following rasters:
#   + strahler stream order dataset (saved as strahler.tif)
#
#   Requirements: GDAL 1.8+, numpy
#
#   This script is written to be run from the command line.
#   You need to specify the chosen stream initiation upstream drainage area threshold (in km^2)
#   for Strahler stream order calculation, the default value is 1km^2
#
#   Ensure terminal is in directory that contains the strahler.py script. Then run as below where
#   (-t) is the chosen drainage area stream initiation threshold and (-o) is overwrite (optional)
#
#   python strahler.py -i C:\path\to\directory\with\files -t 1 -o
#
#   Author: Mark Bernhofen (2019) cn13mvb@leeds.ac.uk
#####################################################################

import argparse
import os
import sys
from datetime import datetime

import numpy as np
from osgeo import gdal


class strahler:

    def __init__(self, inputDir, overwrite, accThresh):

        self.inputDir = inputDir
        if not os.path.exists(self.inputDir):
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
        self.drainOutToDirectionValues = [1, 2, 4, 8, 16, 32, 64, 128]  # cells draining OUT OF cell
        self.riverCellCounter = 0

        # Define input filenames and then read in the data.
        self.dirFile = os.path.join(self.inputDir, 'dir.tif')
        self.dir = self.readData(self.dirFile, saveInfo=True).astype(np.uint8)
        self.accFile = os.path.join(self.inputDir, 'acc.tif')
        self.acc = self.readAccData(self.accFile)
        self.strahlerFile = os.path.join(self.outputDir, "strahler.tif")

        # Get raster info
        self.totalCells = np.count_nonzero(self.acc)

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

        # Create an empty Strahler array that will be filled during the analysis
        self.strahler = np.full(self.dir.shape, 0, dtype=np.int16)


        # Create an empty list of next cells to iterate over. This list will be appended to throuhgout the process
        self.nextIterations = []

        # What is the maximum Strahler order? This will update throughout the process
        self.maxStrahler = 0

        # Begin the analysis at all stream source locations. This part of the script will scan the entire acc (river)
        # array and begin the analysis at stream cells that have no rivers draining into it. These cells are assumed
        # to be river source locations. This first part of the script finds all Strahler order 1 streams.

        for r in range(self.rasterYSize):
            for c in range(self.rasterXSize):
                if self.acc[r, c] != 0:
                    neighbors = self.neighborhood(r, c)
                    # Call the multidrain function which checks how many river cells drain into current cell
                    multiDrain = self.multiDrain(neighbors)
                    if multiDrain == 0: # will be river source
                        dir = self.dir[r, c]
                        if dir == 247:  # undefined
                            continue
                        if dir == 255:  # inland depression
                            continue
                        if dir == 0:    # ocean
                            continue
                        # If no no-data values were encountered then the process for this stream begins
                        self.riverCellCounter += 1
                        self.strahler[r, c] = 1
                        nextCell = self.neighbor(dir, r, c)
                        if nextCell is not None:
                            self.neighborProcess(nextCell, 1)

        # Once all the first order streams have been processed. The analysis will continue for all the remaining
        # river cells in the dataset. Each iteration of this analysis part of the analysis will begin at a confluence
        # location that has been stored in the nextIterations list. And all subsequent confluence locations will be
        # appended to the nextIterations list until all the river cells have been processed.
        while len(self.nextIterations) > 0:
            nextIteration = self.nextIterations.pop(0)
            row = nextIteration[0]
            col = nextIteration[1]
            order = nextIteration[2]
            dir = self.dir[row, col]

            if dir == 247:  # undefined
                continue
            if dir == 255:  # inland depression
                continue
            if dir == 0:    # ocean
                continue

            self.riverCellCounter += 1
            self.strahler[row, col] = order
            nextCell = self.neighbor(dir, row, col)
            if nextCell is not None:
                self.neighborProcess(nextCell, order)

    def multiDrain(self, neighbors):
        '''
        This function checks and returns the value of how many of the surrounding cells drain into the cell
        '''
        multiDrain = 0
        for idx, n in enumerate(neighbors):
            if n is not None:
                pixel = (n[0], n[1], idx)
                # Call the doesItDrain function which returns 1 if the neighboring cell drains into this one
                # and returns 0 if it does not drain.
                multi = self.doesItDrain(pixel)
                multiDrain += multi

        return multiDrain

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

    def neighbor(self, dir, row, col):
        '''
        This function returns the next downstream cell, given the current cell's flow direction and coordinates
        The neighboring elements are labelled as follows:
        5  6  7
        4  x  0
        3  2  1
        this matches the flow directions defined in the previous __init__ function
        '''
        dir = dir
        neighbors = [[row, col + 1], [row + 1, col + 1], [row + 1, col], [row + 1, col - 1], [row, col - 1],
                     [row - 1, col - 1], [row - 1, col], [row - 1, col + 1]]
        dirIndex = self.drainOutToDirectionValues.index(dir)
        neighbor = neighbors[dirIndex]

        if neighbor[0] < 0:
            return None
        if neighbor[1] < 0:
            return None

        return neighbor

    def neighborProcess(self, nextCell, order):
        '''
        This function processes the current pixel and checks the neighboring pixels for the one that drains
        into it and assigns the current Strahler stream order to that pixel. This process will continue until a stream
        confluence location has been reached.
        '''
        q = [nextCell]
        while len(q) > 0:
            neighbor = q.pop(0)
            row = neighbor[0]
            col = neighbor[1]
            order = order
            dir = self.dir[row, col]

            if dir == 247:  # undefined
                continue
            if dir == 255:  # inland depression
                continue
            if dir == 0:    # ocean
                continue

            # how many river cells drain into this cell?
            neighbors = self.neighborhood(row, col)
            multiDrain = self.multiDrain(neighbors)
            # if more than one river cell drains into this cell then it is a confluence
            if multiDrain > 1:
                # Have any of the neighbors already been processed?
                existingNeighbors = self.countExistingNeighbors(row, col)
                # If all the draining river cells have already been processed. Then the Strahler stream order
                # of the next cell downstream needs to be determined. This will either be the largest stream order or
                # in the case of > 1 largest stream order cells the next downstream cell will have order largest
                # previous + 1.
                if multiDrain == existingNeighbors:
                    # Call the dominantStreamOrder function which determines the next stream order based on neighboring
                    # draining cells.
                    streamOrder = self.dominantStreamOrder(row, col)
                    self.nextIterations.append([row, col, streamOrder])
                    if streamOrder > self.maxStrahler:
                        self.maxStrahler = streamOrder
                    return
                else:
                    return

            self.strahler[row, col] = order
            self.riverCellCounter += 1

            nextNeighbor = self.neighbor(dir, row, col)
            if nextNeighbor is not None:
                q.append(nextNeighbor)

    def countExistingNeighbors(self, row, col):
        '''
        This function counts the number of surrounding DRAINING cells that have already been processed by the code
        and returns that number.
        '''
        neighbors = self.neighborhood(row, col)
        existingNeighbors = 0
        for idx, n in enumerate(neighbors):
            if n is not None:
                pixel = (n[0], n[1], idx)
                r = pixel[0]
                c = pixel[1]
                if self.strahler[r, c] > 0 and self.doesItDrain(pixel) == 1:
                    existingNeighbors += 1

        return existingNeighbors

    def dominantStreamOrder(self, row, col):
        '''
        This function finds the surrounding stream orders draining into the cell and returns the value for the
        dominant one (either the maximum value) or n + 1 of the two largest draining stream orders
        '''
        neighbors = self.neighborhood(row, col)
        orders = []
        for idx, n in enumerate(neighbors):
            if n is not None:
                pixel = (n[0], n[1], idx)
                r = pixel[0]
                c = pixel[1]
                if self.strahler[r, c] > 0 and self.doesItDrain(pixel) == 1:
                    orders.append(self.strahler[r, c])

        # Check if there are any duplicate stream orders draining into this confluence location
        duplicates = [x for x in orders if orders.count(x) > 1]
        # if there are no duplicates, then the stream order will just be the maximum order draining into the cell
        if not duplicates:
            maxe = max(orders)
            order = maxe
        # if duplicates do exist and they are also the largest order. Then new order will be order + 1.
        else:
            maxd = max(duplicates)
            maxe = max(orders)
            if maxd >= maxe:
                order = maxd + 1
            else:
                order = maxe

        return order

    def doesItDrain(self, neighbor):
        '''
        This function checks the neighboring cell and if it drains into the river cell it returns 1
        and if it doesn't it returns 0.
        '''
        row = neighbor[0]
        col = neighbor[1]
        index = neighbor[2]
        dir = self.dir[row, col]

        if dir == 247:
            return 0
        if dir == 255:
            return 0
        if self.acc[row, col] > 0 and dir == self.drainInToDirectionValues[index]:
            return 1
        else:
            return 0


    def writeTifOutput(self):
        print(str(datetime.now()), 'Saving GeoTiffs...')
        print('Total river cells processed', self.riverCellCounter)
        print('Total accumulation cells that should have been processed', self.totalCells)

        # Write text file with threshold information
        fLoc = os.path.join(self.resultsDir, "read_me.txt")
        f = open(fLoc, "w+")
        f.write("Stream initiation upstream drainage area threshold: " + str(self.accThresh) + "km^2" + "\n")
        f.write("Maximum Strahler order: " + str(self.maxStrahler))
        f.close()

        driver = gdal.GetDriverByName("GTiff")
        dst = driver.Create(self.strahlerFile, self.rasterXSize, self.rasterYSize, 1, gdal.GDT_Int16)
        band = dst.GetRasterBand(1)
        band.WriteArray(self.strahler, 0, 0)

        projection = self.dataInfo.GetProjection()
        geotransform = self.dataInfo.GetGeoTransform()

        dst.SetGeoTransform(geotransform)
        dst.SetProjection(projection)
        dst.FlushCache()

        dst = None
        self.dataInfo = None

        return

################################################################################
# Main
#
# strahler.py -i 'input_directory' -t 1 -o

if __name__ == "__main__":
    version_num = int(gdal.VersionInfo('VERSION_NUM'))
    if version_num < 1800: # because of GetGeoTransform(can_return_null)
        print('ERROR: Python bindings of GDAL 1.8.0 or later required')
        sys.exit(-1)

    parser = argparse.ArgumentParser(description='Generate Strahler Raster')
    apg_input = parser.add_argument_group('Input')
    apg_input.add_argument("-i", "--inputdir", nargs='?',
                           help="Filepath to directory containing all 3 required input files")
    apg_input.add_argument("-o", "--overwrite", action='store_true',
                           help="Will overwrite any existing files found in hand directory")
    apg_input.add_argument("-t", "--threshold", default=1, nargs='?',
                           help="Drainage area threshold to consider [kilometres squared]")
    options = parser.parse_args()

    S = strahler(inputDir=options.inputdir, overwrite=options.overwrite, accThresh=options.threshold)

    if not os.path.isfile(S.strahlerFile) or options.overwrite:
        S.mainProcess()
        S.writeTifOutput()

    else:
        print('strahler.tif exists', S.strahlerFile)

################################################################################