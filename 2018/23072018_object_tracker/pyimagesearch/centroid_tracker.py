# import the necessary packages
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

class CentroidTracker():
    def __init__(self, maxDisappeared=50):
        # initialize the next unique object ID alogn with two ordered 
        # dictionaries used to keep track of mapping given object ID
        # to its centroid and number of consecutive frames it has
        # been marked as "dissapeared", respectively
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.dissapeared = OrderedDict()

        # store the number of maximum consecutive frames a given
        # object is allowed to be marked as "dissapeared" until we
        # need to deregister the object from tracking
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        # when registering an object we use the next available object ID
        # to store the centroid
        self.objects[self.nextObjectID] = centroid
        self.dissapeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        # to deregister and object ID we delete the object ID from
        # both of our respective dictionaries
        del self.objects[objectID]
        del self.dissapeared[objectID]

    def update(self, rects):
        # assumed rects form: (startX, startY, endX, endY)
        # check to see if the list of input bounding box rectangles is empty
        if len(rects) == 0:
            # loop over any existing tracked objects and mark them as disappeared
            for objectID in self.dissapeared.keys():
                self.dissapeared[objectID] += 1

                # if we have reached a maximum number of consecutive frames
                # where a given object has been marked as missing, deregister it
                if self.dissapeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            # return early as there are no centroids or tracking info to update
            return self.objects

        # initialize an array of inputs centroids for the current frame
        inputCentroids = np.zeros((len(rects), 2), dtype="int")

        # loop over the bouding box rectangles
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            # use the bouding box coordinates to derive the centroid
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        # if we are currently not tracking any objects take the input
        # centroids and register each of them
        # otherwise, we are currently tracking objects so we need to try
        # match the input centroids to existing object centroids
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            # grab the set of object IDs and corresponding centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            # compute the distance between each pair of object centroids
            # and input centroids, respectively --our goal will be to match
            # an input centroid to and existing object centroid
            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            # in order to perform this matching we must (1) find the smallest 
            # value in each row and then (2) sort the row indexes based on their 
            # minimum values so that the row with the smallest value is at the *front* 
            # of the index list
            rows = D.min(axis=1).argsort()

            # next, we perform a similar process on the columns by finding the smallest
            # value in each column and then sorting the previously computed row index list
            cols = D.argmin(axis=1)[rows]

            # in order to determine if we need to update, register, or deregister and object
            # we need to keep track of which of the rows and column indexes we have already examined
            usedRows = set()
            usedCols = set()

            # loop over the combination of the (row, column) index tuples
            for (row, col) in zip(rows, cols):
                # if we have already examined either the row or column value
                # before, ignore it val
                if row in usedRows or col in usedCols:
                    continue

                # otherwise, grab the object ID for the current row, set its centroid, and reset the
                # disappeared counter
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.dissapeared[objectID] = 0

                # indicate that we have examined each of the row and column indexes
                usedRows.add(row)
                usedCols.add(col)

            # compute both the row and column index we have NOT yet examined
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)    
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # in the event that the number of object centroids is equal or greater than 
            # the number of input centroids we need to check and see if some of these objects
            # have potentially disappeared
            # otherwise, if the number of input centroids is greater than the number of existing object 
            # centroids we need to register each new input centroid as a trackable object
            if D.shape[0] >= D.shape[1]:
                # loop over the unused row indexes
                for row in unusedRows:
                    # grab the object ID for the corresponding row index and increment the disappeared counter
                    objectID = objectIDs[row]
                    self.dissapeared[objectID] += 1

                    # check to see if the number of consecutive frames, the object has been marked "dissapeared"
                    # for warrants deregistering the object
                    if self.dissapeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])
    
        # return the set of trackable objects
        return self.objects
