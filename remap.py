import itertools
import numpy
import logging

def vec3(*args):
	if len(args) == 1: args = args[0]
	return numpy.array(args,dtype=numpy.float64)

def dot(u, v):
	return u[0]*v[0] + u[1]*v[1] + u[2]*v[2]

def square(v):
	return v[0]**2 + v[1]**2 + v[2]**2

def norm(v):
	return numpy.sqrt(square(v))

def triple_scalar_product(u, v, w):
	return u[0]*(v[1]*w[2] - v[2]*w[1]) + u[1]*(v[2]*w[0] - v[0]*w[2]) + u[2]*(v[0]*w[1] - v[1]*w[0])


class Plane(object):
	def __init__(self, p, n):
		self.n = n
		self.d = -dot(p,n)

	@property
	def normal(self):
		return self.n/norm(self.n)

	def test(self, pos):
		"""Compare a point to a plane.  Return value is positive, negative, or
		zero depending on whether the point lies above, below, or on the plane."""
		return dot(self.n,pos) + self.d


class Cell(object):
	def __init__(self, ipos=(0,0,0)):
		self.ipos = vec3(ipos)
		self.faces = []

	def contains(self, pos):
		mask = numpy.ones_like(pos[0],dtype=numpy.bool_)
		for face in self.faces:
			mask &= face.test(pos) >= 0
		return mask
	
	def __repr__(self):
		return "Cell at %s with %d non-trivial planes" % (self.ipos, len(self.faces))

	
def test_unit_cube(P):
	"""Return +1, 0, or -1 if the unit cube is above, below, or intersecting the plane."""
	pos = vec3([(0,0,0), (0,0,1), (0,1,0), (0,1,1), (1,0,0), (1,0,1), (1,1,0), (1,1,1)]).T
	s = P.test(pos)
	above = (s>0).any()
	below = (s<0).any()
	return int(above) - int(below)


class Cuboid(object):
	"""Cuboid remapping class."""
	
	logger = logging.getLogger("Cuboid")

	def __init__(self, u1=(1,0,0), u2=(0,1,0), u3=(0,0,1)):
		"""Initialize by passing a 3x3 invertible integer matrix."""
		u1 = vec3(u1)
		u2 = vec3(u2)
		u3 = vec3(u3)
		if triple_scalar_product(u1, u2, u3) != 1.:
			raise ValueError("Invalid lattice vectors: u1 = %s, u2 = %s, u3 = %s" % (u1,u2,u3))
			self.e1 = vec3(1,0,0)
			self.e2 = vec3(0,1,0)
			self.e3 = vec3(0,0,1)
		else:
			s1 = square(u1)
			s2 = square(u2)
			d12 = dot(u1, u2)
			d23 = dot(u2, u3)
			d13 = dot(u1, u3)
			alpha = -d12/s1
			gamma = -(alpha*d13 + d23)/(alpha*d12 + s2)
			beta = -(d13 + gamma*d12)/s1
			self.e1 = u1
			self.e2 = u2 + alpha*u1
			self.e3 = u3 + beta*u1 + gamma*u2

		self.logger.info("e1 = %s" % self.e1)
		self.logger.info("e2 = %s" % self.e2)
		self.logger.info("e3 = %s" % self.e3)

		self.L1 = norm(self.e1)
		self.L2 = norm(self.e2)
		self.L3 = norm(self.e3)
		self.n1 = self.e1/self.L1
		self.n2 = self.e2/self.L2
		self.n3 = self.e3/self.L3
		self.cells = []

		v0 = vec3(0,0,0)
		self.v = [v0,
				  v0 + self.e3,
				  v0 + self.e2,
				  v0 + self.e2 + self.e3,
				  v0 + self.e1,
				  v0 + self.e1 + self.e3,
				  v0 + self.e1 + self.e2,
				  v0 + self.e1 + self.e2 + self.e3]

		# Compute bounding box of cuboid
		vmin = numpy.min(self.v,axis=0)
		vmax = numpy.max(self.v,axis=0)

		# Extend to nearest integer coordinates
		iposmin = numpy.floor(vmin).astype(int)
		iposmax = numpy.ceil(vmax).astype(int)
		self.logger.info("min - max: %s %s" % (iposmin,iposmax))
		
		# Determine which cells (and which faces within those cells) are non-trivial
		iranges = [numpy.arange(min_,max_) for min_,max_ in zip(iposmin,iposmax)]
		for ipos in itertools.product(*iranges):
			shift = -vec3(ipos)
			faces = [Plane(self.v[0] + shift, +self.n1),
					Plane(self.v[4] + shift, -self.n1),
					Plane(self.v[0] + shift, +self.n2),
					Plane(self.v[2] + shift, -self.n2),
					Plane(self.v[0] + shift, +self.n3),
					Plane(self.v[1] + shift, -self.n3)]
			cell = Cell(ipos)
			skipcell = False
			for face in faces:
				r = test_unit_cube(face)
				if r == 1:
					# Unit cube is completely above this plane; this cell is empty
					continue
				elif r == 0:
					# Unit cube intersects this plane; keep track of it
					cell.faces.append(face)
				elif r == -1:
					skipcell = True
					break
			if skipcell or len(cell.faces) == 0:
				self.logger.info("Skipping cell at %s" % str(ipos))
				continue
			else:
				self.cells.append(cell)
				self.logger.info("Adding cell at %s" % str(ipos))

		# For the identity remapping, use exactly one cell
		if len(self.cells) == 0:
			self.cells.append(Cell())

		# Print the full list of cells
		self.logger.info("%d non-empty cells" % len(self.cells))
		for cell in self.cells:
			self.logger.info(str(cell))

	def transform(self, pos):
		newpos = vec3(pos)
		masktot = numpy.zeros((newpos.shape[-1]),dtype=numpy.bool_)
		for cell in self.cells:
			mask = cell.contains(pos)
			masktot |= mask
			newpos[:,mask] += cell.ipos[:,None]
		if not masktot.all():
			raise RuntimeError("Elements not contained in any cell")
		return vec3(dot(newpos, self.n1), dot(newpos, self.n2), dot(newpos, self.n3))

	def inverse_transform(self, pos):
		newpos = pos[0]*self.n1 + pos[1]*self.n2 + pos[2]*self.n3
		return numpy.fmod(newpos, 1) + (newpos < 0)

