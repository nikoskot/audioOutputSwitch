import queue
import numpy as np


def changeReferencePoint(pointCoords, newRefPointCoords):
    return [pointCoords[0] - newRefPointCoords[0], pointCoords[1] - newRefPointCoords[1]]


def normalizeCoords(pointCoords, normFactorX, normFactorY):
    return [pointCoords[0] / normFactorX, pointCoords[1] / normFactorY]


def insertAtEndOfQueue(queue, value):
    if queue.full():
        removeFromFrontOfQueue(queue)
        queue.put(value)
    else:
        queue.put(value)
    return queue


def removeFromFrontOfQueue(queue):
    if queue.empty():
        return queue
    else:
        return queue.get()


def clearQueue():
    return queue.Queue(maxsize=100)


def chechForSwitch(buffer):
    buffer = np.array(list(buffer.queue))
    size = buffer.size
    if size > 10:
        mostFreqOfFirstHalf = np.argmax(np.bincount(buffer[0:(size//2)]))
        mostFreqOfSecondHalf = np.argmax(np.bincount(buffer[((size//2)+1):]))
        if (mostFreqOfFirstHalf == 1) & (mostFreqOfSecondHalf == 2):
            return True
    return False
