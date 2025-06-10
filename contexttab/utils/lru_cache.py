# SPDX-FileCopyrightText: 2025 SAP SE
#
# SPDX-License-Identifier: Apache-2.0

class LRU_Cache:
    """
    This is just a remake of functools.lru_cache without the function call.
    """
    full = False
    PREV, NEXT, KEY, RESULT = 0, 1, 2, 3  # names for the link fields
    cache = {}
    hits = misses = 0
    root = []  # root of the circular doubly linked list
    root[:] = [root, root, None, None]  # initialize by pointing to self

    def __init__(self, max_size=1000):
        self.max_size = max_size

    def __getitem__(self, key):
        link = self.cache.get(key)
        if link is None:
            self.misses += 1
            return None
        # Move the link to the front of the circular queue
        link_prev, link_next, _, result = link
        link_prev[self.NEXT] = link_next
        link_next[self.PREV] = link_prev
        last = self.root[self.PREV]
        last[self.NEXT] = self.root[self.PREV] = link
        link[self.PREV] = last
        link[self.NEXT] = self.root
        self.hits += 1
        return result

    def __setitem__(self, key, value):
        if key in self.cache:
            # We would need to update the link, but for our usage
            # we only call set after get, so no real need.
            return

        if self.full:
            # Use the old root to store the new key and result.
            oldroot = self.root
            oldroot[self.KEY] = key
            oldroot[self.RESULT] = value
            # Empty the oldest link and make it the new root.
            # Keep a reference to the old key and old result to
            # prevent their ref counts from going to zero during the
            # update. That will prevent potentially arbitrary object
            # clean-up code (i.e. __del__) from running while we're
            # still adjusting the links.
            self.root = oldroot[self.NEXT]
            oldkey = self.root[self.KEY]
            oldresult = self.root[self.RESULT]
            self.root[self.KEY] = self.root[self.RESULT] = None
            # Now update the cache dictionary.
            del self.cache[oldkey]
            # Save the potentially reentrant cache[key] assignment
            # for last, after the root and links have been put in
            # a consistent state.
            self.cache[key] = oldroot
        else:
            # Put result in a new link at the front of the queue.
            last = self.root[self.PREV]
            link = [last, self.root, key, value]
            last[self.NEXT] = self.root[self.PREV] = self.cache[key] = link
            # Use the cache_len bound method instead of the len() function
            # which could potentially be wrapped in an lru_cache itself.
            self.full = (self.cache.__len__() >= self.max_size)
