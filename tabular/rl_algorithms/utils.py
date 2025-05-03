from queue import PriorityQueue

class MaxPriorityQueue:
    """
    Maximum Priority Queue without replacement (no repeated elements, only the data with highest priority is kept).
    len is the number of useful data stored in the data structure, but len is not real, more memory may be stored than then number of data returned by len.
    qsize returns the true number of data stored, but they may be duplicated.
    """
    def __init__(self):
        self._pq = PriorityQueue()
        self._dict = {}

    def put(self, priority, data):
        if data in self._dict:
            old_priority = self._dict[data]
            if priority > old_priority:
                self._dict[data] = priority
                self._pq.put_nowait((-1.0 * priority, data))
        else:
            self._dict[data] = priority
            self._pq.put_nowait((-1.0 * priority, data))

    def get(self):
        while True:
            priority, data = self._pq.get_nowait()
            priority = -1.0 * priority

            if data in self._dict and priority == self._dict[data]:
                break
        
        self._dict.pop(data)

        if self.empty():
            self._pq = PriorityQueue()

        return priority, data

    def empty(self):
        return len(self._dict) == 0

    def qsize(self):
        return self._pq.qsize()
    
    def __len__(self):
        return len(self._dict)