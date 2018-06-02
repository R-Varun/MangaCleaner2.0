import skimage
import numpy as np
import PIL
import itertools


def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	stepx = 0
	stepy = 0

	if type(stepSize) == type((1,)):
		stepx = stepSize[0]
		stepy = stepSize[1]
	else:
		stepx = stepSize
		stepy = stepSize
	for y in range(0, image.shape[0] - windowSize[0] + 1, stepy):
		for x in range(0, image.shape[1] - windowSize[1] + 1, stepx):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


import itertools


class Rectangle:
	def intersection(self, other):
		a, b = self, other
		x1 = max(min(a.x1, a.x2), min(b.x1, b.x2))
		y1 = max(min(a.y1, a.y2), min(b.y1, b.y2))
		x2 = min(max(a.x1, a.x2), max(b.x1, b.x2))
		y2 = min(max(a.y1, a.y2), max(b.y1, b.y2))
		if x1 < x2 and y1 < y2:
			return type(self)(x1, y1, x2, y2)

	__and__ = intersection

	def difference(self, other):
		inter = self & other
		if not inter:
			yield self
			return
		xs = {self.x1, self.x2}
		ys = {self.y1, self.y2}
		if self.x1 < other.x1 < self.x2: xs.add(other.x1)
		if self.x1 < other.x2 < self.x2: xs.add(other.x2)
		if self.y1 < other.y1 < self.y2: ys.add(other.y1)
		if self.y1 < other.y2 < self.y2: ys.add(other.y2)
		for (x1, x2), (y1, y2) in itertools.product(
				pairwise(sorted(xs)), pairwise(sorted(ys))
		):
			rect = type(self)(x1, y1, x2, y2)
			if rect != inter:
				yield rect

	__sub__ = difference

	def __init__(self, x1, y1, x2, y2):
		if x1 > x2 or y1 > y2:
			raise ValueError("Coordinates are invalid")
		self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
		self.area = (x2 - x1) * (y2 - y1)

	def __iter__(self):
		yield self.x1
		yield self.y1
		yield self.x2
		yield self.y2

	def __eq__(self, other):
		return isinstance(other, Rectangle) and tuple(self) == tuple(other)

	def __ne__(self, other):
		return not (self == other)

	def __repr__(self):
		return type(self).__name__ + repr(tuple(self))

	def intersects(self, other):
		return self.x1 < other.x2 and self.x2 > other.x1 and \
			   self.y1 < other.y2 and self.y2 > other.y1


def pairwise(iterable):
	# https://docs.python.org/dev/library/itertools.html#recipes
	a, b = itertools.tee(iterable)
	next(b, None)
	return zip(a, b)
