import logging
import collections
import os
import scipy
from scipy import interpolate,optimize,stats,constants
from remap import Cuboid

def save(path,*args,**kwargs):
	mkdir(os.path.dirname(path))
	logger.info('Saving to {}.'.format(path))
	scipy.save(path,*args,**kwargs)

def rotation_matrix_from_vectors(a,b):

	a = scipy.asarray(a)
	b = scipy.asarray(b)
	a /= scipy.linalg.norm(a)
	b /= scipy.linalg.norm(b)
	v = scipy.cross(a,b)
	c = scipy.dot(a,b)
	s = scipy.linalg.norm(v)
	I = scipy.identity(3,dtype='f8')
	k = scipy.array([[0., -v[2], v[1]],[v[2], 0., -v[0]],[-v[1], v[0], 0.]])
	if s == 0.: return I
	return I + k + scipy.matmul(k,k) * ((1.-c)/(s**2))

def distance(position):
	return scipy.sqrt((position**2).sum(axis=-1))
		
def cartesian_to_sky(position,wrap=True,degree=True):
	"""Transform cartesian coordinates into distance, RA, Dec.

	Parameters
	----------
	position : array of shape (N,3)
		position in cartesian coordinates.
	wrap : bool, optional
		whether to wrap ra into [0,2*pi]
	degree : bool, optional
		whether RA, Dec are in degree (True) or radian (False).

	Returns
	-------
	dist : array
		distance.
	ra : array
		RA.
	dec : array
		Dec.

	"""
	dist = distance(position)
	ra = scipy.arctan2(position[:,1],position[:,0])
	if wrap: ra %= 2.*constants.pi
	dec = scipy.arcsin(position[:,2]/dist)
	if degree: return dist,ra/constants.degree,dec/constants.degree
	return dist,ra,dec

def sky_to_cartesian(dist,ra,dec,degree=True,dtype=None):
	"""Transform distance, RA, Dec into cartesian coordinates.

	Parameters
	----------
	dist : array
		distance.
	ra : array
		RA.
	dec : array
		Dec.
	degree : bool
		whether RA, Dec are in degree (True) or radian (False).
	dtype : dtype, optional
		return array dtype.

	Returns
	-------
	position : array
		position in cartesian coordinates; of shape (len(dist),3).

	"""
	conversion = 1.
	if degree: conversion = constants.degree
	position = [None]*3
	cos_dec = scipy.cos(dec*conversion)
	position[0] = cos_dec*scipy.cos(ra*conversion)
	position[1] = cos_dec*scipy.sin(ra*conversion)
	position[2] = scipy.sin(dec*conversion)
	return (dist*scipy.asarray(position,dtype=dtype)).T

def cutsky_to_box(drange,rarange,decrange):
	"""Translation along x and rotation about z and y."""
	if rarange[0] > rarange[1]: rarange[0] -= 360.
	deltara = abs(rarange[1]-rarange[0])/2.*constants.degree
	deltadec = abs(decrange[1]-decrange[0])/2.*constants.degree
	boxsize = scipy.empty((3),dtype=scipy.float64)
	boxsize[1] = 2.*drange[1]*scipy.sin(deltara)
	boxsize[2] = 2.*drange[1]*scipy.sin(deltadec)
	boxsize[0] = drange[1] - drange[0]*min(scipy.cos(deltara),scipy.cos(deltadec))
	operations = [{'method':'translate_along_axis','kwargs':{'axis':'x','translate':drange[1]-boxsize[0]/2.}}]
	operations += [{'method':'rotate_about_origin_axis','kwargs':{'axis':'y','angle':(decrange[0]+decrange[1])/2.}}]
	operations += [{'method':'rotate_about_origin_axis','kwargs':{'axis':'z','angle':(rarange[0]+rarange[1])/2.}}]
	return boxsize,operations

def box_to_cutsky(boxsize,dmax):
	deltara = scipy.arcsin(boxsize[1]/2./dmax)
	deltadec = scipy.arcsin(boxsize[2]/2./dmax)
	dmin = (dmax-boxsize[0])/min(scipy.cos(deltara),scipy.cos(deltadec))
	return deltara*2./constants.degree,deltadec*2./constants.degree,dmin

def cartesian(arrays, out=None):
	"""Generate a cartesian product of input arrays.

	Parameters
	----------
	arrays : list of array-like
		1-D arrays to form the cartesian product of.
	out : ndarray
		Array to place the cartesian product in.

	Returns
	-------
	out : ndarray
		2-D array of shape (M, len(arrays)) containing cartesian products
		formed of input arrays.

	Examples
	--------
	>>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
	array([[1, 4, 6],
		   [1, 4, 7],
		   [1, 5, 6],
		   [1, 5, 7],
		   [2, 4, 6],
		   [2, 4, 7],
		   [2, 5, 6],
		   [2, 5, 7],
		   [3, 4, 6],
		   [3, 4, 7],
		   [3, 5, 6],
		   [3, 5, 7]])

	"""
	arrays = [scipy.asarray(x) for x in arrays]
	dtype = arrays[0].dtype

	n = scipy.prod([x.size for x in arrays])
	if out is None:
		out = scipy.zeros([n, len(arrays)], dtype=dtype)

	m = n / arrays[0].size
	out[:,0] = scipy.repeat(arrays[0], m)
	if arrays[1:]:
		cartesian(arrays[1:], out=out[0:m,1:])
		for j in xrange(1, arrays[0].size):
			out[j*m:(j+1)*m,1:] = out[0:m,1:]
	return out

class Catalogue(object):

	logger = logging.getLogger('Catalogue')
	
	def __init__(self,columns={},fields=None,**attrs):
		self.columns = {}
		if fields is None: fields = columns.keys()
		for key in fields:
			self.columns[key] = scipy.array(columns[key])
		self.attrs = attrs
	
	@classmethod
	def from_fits(cls,path,ext=1,**kwargs):
		from astropy.io import fits
		self = cls()
		self.logger.info('Loading catalogue {}.'.format(path))
		hdulist = fits.open(path,mode='readonly',memmap=True)
		columns = hdulist[ext].columns
		
		return cls(columns=hdulist[ext].data,fields=columns.names,**kwargs)
	
	@classmethod
	def from_nbodykit(cls,catalogue,fields=None,allgather=True,**kwargs):
		if fields is None:
			columns = {key: catalogue[key].compute() for key in catalogue}
		else:
			columns = {key: catalogue[key].compute() for key in fields}
		if allgather:
			columns = {key: scipy.concatenate(catalogue.comm.allgather(columns[key])) for key in columns}
		attrs = getattr(catalogue,'attrs',{})
		attrs.update(kwargs)

		return cls(columns=columns,**attrs)
		
	def to_nbodykit(self,fields=None):
	
		from nbodykit.base.catalog import CatalogSource
		from nbodykit import CurrentMPIComm
	
		comm = CurrentMPIComm.get()
		if comm.rank == 0:
			source = self
		else:
			source = None
		source = comm.bcast(source)

		# compute the size
		start = comm.rank * source.size // comm.size
		end = (comm.rank  + 1) * source.size // comm.size

		new = object.__new__(CatalogSource)
		new._size = end - start
		CatalogSource.__init__(new,comm=comm)
		for key in source.fields:
			new[key] = new.make_column(source[key])[start:end]
		new.attrs.update(source.attrs)

		return new
			
	def to_fits(self,path,fields=None,remove=[]):
		from astropy.table import Table
		if fields is None: fields = self.fields
		if remove:
			for rm in remove: fields.remove(rm)
		table = Table([self[field] for field in fields],names=fields)
		self.logger.info('Saving catalogue to {}.'.format(path))
		table.write(path,format='fits',overwrite=True)
		
	def shuffle(self,fields=None,seed=None):
		if fields is None: fields = self.fields
		rng = scipy.random.RandomState(seed=seed)
		indices = self.indices()
		rng.shuffle(indices)
		for key in fields: self[key] = self[key][indices]
	
	def indices(self):
		return scipy.arange(self.size)
	
	def getstate(self,fields=None):
		return {'columns':self.as_dict(fields),'attrs':self.attrs}

	def setstate(self,state):
		self.__dict__.update(state)

	@classmethod
	def loadstate(cls,state):
		self = object.__new__(cls)
		self.setstate(state)
		return self
	
	def save(self,save,keep=None):
		pathdir = os.path.dirname(save)
		mkdir(pathdir)
		self.logger.info('Saving {} to {}.'.format(self.__class__.__name__,save))
		scipy.save(save,self.getstate(keep))
	
	@classmethod
	def load(cls,path):
		state = {}
		try:
			state = scipy.load(path)[()]
		except IOError:
			raise IOError('Invalid path: {}.'.format(path))
		cls.logger.info('Loading {}: {}.'.format(cls.__name__,path))
		return cls.loadstate(state)

	def copy(self):
		return self.__class__.loadstate(self.__dict__)

	def deepcopy(self):
		import copy
		return copy.deepcopy(self)
	
	def slice(self,islice=0,nslices=1):
		size = len(self)
		return self[islice*size//nslices:(islice+1)*size//nslices]
		
	def distance(self,position='Position',axis=-1):
		return scipy.sqrt((self[position]**2).sum(axis=axis))
		
	def box(self,position='Position',axis=-1):
		axis = 0 if axis==-1 else -1
		return (self[position].min(axis=axis),self[position].max(axis=axis))
	
	def boxsize(self,position='Position',axis=-1):
		lbox = scipy.diff(self.box(position=position,axis=axis),axis=0)[0]
		return scipy.sqrt((lbox**2).sum(axis=0))
		
	def __getitem__(self,name):
		if isinstance(name,list) and isinstance(name[0],(str,unicode)):
			return [self[name_] for name_ in name]
		if isinstance(name,(str,unicode)):
			if name in self.fields:
				return self.columns[name]
			else:
				raise KeyError('There is no field {} in the data.'.format(name))
		else:
			new = self.deepcopy()
			new.columns = {field:self.columns[field][name] for field in self.fields}
			return new
	
	def __setitem__(self,name,item):
		if isinstance(name,list) and isinstance(name[0],(str,unicode)):
			for name_,item_ in zip(name,item):
				self.data[name_] = item_
		if isinstance(name,(str,unicode)):
			self.columns[name] = item
		else:
			for key in self.fields:
				self.columns[key][name] = item

	def __delitem__(self,name):
		del self.columns[name]

	def	__contains__(self,name):
		return name in self.columns

	def __iter__(self):
		for field in self.columns:
			yield field
	
	def __str__(self):
		return str(self.columns)

	def __len__(self):
		return len(self[self.fields[0]])

	@property
	def size(self):
		return len(self)
	
	def zeros(self,dtype=scipy.float64):
		return scipy.zeros(len(self),dtype=dtype)
	
	def ones(self,dtype=scipy.float64):
		return scipy.ones(len(self),dtype=dtype)
	
	def falses(self):
		return self.zeros(dtype=scipy.bool_)
	
	def trues(self):
		return self.ones(dtype=scipy.bool_)
	
	def nans(self):
		return self.ones()*scipy.nan
	
	@property
	def fields(self):
		return self.columns.keys()
		
	def remove(self,name):
		del self.columns[name]
		
	def __radd__(self,other):
	
		if other == 0: return self
		else: return self.__add__(other)

	def __add__(self,other):

		columns = {}
		fields = [field for field in self.fields if field in other.fields]
		for field in fields:
			columns[field] = scipy.concatenate([self[field],other[field]],axis=0)

		import copy
		attrs = copy.deepcopy(self.attrs)
		attrs.update(copy.deepcopy(other.attrs))

		new = self.copy()
		new.columns = columns
		new.attrs = attrs

		return new

	def as_dict(self,fields=None):
		if fields is None: fields = self.fields
		return {field:self[field] for field in fields}

class SurveyCatalogue(Catalogue):

	logger = logging.getLogger('SurveyCatalogue')

	def __init__(self,columns={},fields=None,BoxSize=1.,BoxCenter=0.,Position='Position',Velocity='Velocity',**attrs):
	
		super(SurveyCatalogue,self).__init__(columns=columns,fields=fields,Position=Position,Velocity=Velocity,**attrs)
		self.BoxSize = scipy.empty((3),dtype=scipy.float64)
		self.BoxSize[:] = BoxSize
		self._boxcenter = scipy.empty((3),dtype=scipy.float64)
		self._boxcenter[:] = BoxCenter
		self._rotation = scipy.eye(3,dtype=scipy.float64)
		self._translation =  self._boxcenter.copy()
		self._compute_position = True
		self._compute_velocity = True
		
	@property
	def Position(self):
		if self._compute_position:
			self._position = scipy.tensordot(self[self.attrs['Position']]-self._boxcenter,self._rotation,axes=((1,),(1,))) + self._translation
		self._compute_position = False
		return self._position

	@property
	def Velocity(self):
		if self._compute_velocity:
			self._velocity = scipy.tensordot(self[self.attrs['Velocity']],self._rotation,axes=((1,),(1,)))
		self._compute_velocity = False
		return self._velocity
			
	def VelocityOffset(self,z=0.,E=1.,cosmo=None):
		if cosmo is not None:
			E = self.cosmo.efunc(z)
		return self.Velocity*(1.+z)/(100.*E)
			
	def rotate_about_origin_axis(self,axis=0,angle=0.,degree=True):
		if degree: angle *= constants.degree
		if not isinstance(axis,int): axis = 'xyz'.index(axis)
		c,s = scipy.cos(angle),scipy.sin(angle)
		if axis==0: matrix = [[1.,0.,0.],[0.,c,-s],[0.,s,c]]
		if axis==1: matrix = [[c,0.,-s],[0,1.,0.],[s,0.,c]]
		if axis==2: matrix = [[c,-s,0],[s,c,0],[0.,0,1.]]
		matrix = scipy.asarray(matrix)
		self._rotation = matrix.dot(self._rotation)
		self._translation = matrix.dot(self._translation)
		self._compute_position = True
		self._compute_velocity = True
	
	def rotate_about_center_axis(self,axis=0,angle=0.,degree=True):
		if degree: angle *= constants.degree
		if not isinstance(axis,int): axis = 'xyz'.index(axis)
		c,s = scipy.cos(angle),scipy.sin(angle)
		if axis==0: matrix = [[1.,0.,0.],[0.,c,-s],[0.,s,c]]
		if axis==1: matrix = [[c,0.,s],[0,1.,0.],[-s,0.,c]]
		if axis==2: matrix = [[c,-s,0],[s,c,0],[0.,0,1.]]
		matrix = scipy.asarray(matrix)
		self._rotation = matrix.dot(self._rotation)
		self._compute_position = True
		self._compute_velocity = True
		
	def rotate_about_origin(self,angles=[],degree=True):
		assert len(angles) <= 3
		for axis,angle in enumerate(angles):
			self.rotate_about_origin_axis(axis,angle,degree=degree)
	
	def rotate_about_center(self,angles=[],degree=True):
		assert len(angles) <= 3
		for axis,angle in enumerate(angles):
			self.rotate_about_center_axis(axis,angle,degree=degree)
			
	def translate(self,translate=0.):
		shift = scipy.empty((3),dtype=scipy.float64)
		shift[:] = translate
		self._translation += shift
		self._compute_position = True
	
	def translate_along_axis(self,axis=0,translate=0.):
		if not isinstance(axis,int): axis = 'xyz'.index(axis)
		self._translation[axis] += translate
		self._compute_position = True
	
	def reset_rotate_about_center(self):
		self._rotation = scipy.eye(self._rotation.shape[0],dtype=self._rotation.dtype)
		self._compute_position = True
		self._compute_velocity = True
		
	def reset_rotate_about_origin(self):
		self._translation = self._rotation.T.dot(self._translation)
		self._rotation = scipy.eye(self._rotation.shape[0],dtype=self._rotation.dtype)
		self._compute_position = True
		self._compute_velocity = True
		
	def reset_translate(self):
		self._translation[:] = self._boxcenter[:]
		self._compute_position = True
	
	def recenter(self):
		self._translation[:] = 0.
		self._compute_position = True

	def recenter_position(self,position):
		return scipy.tensordot(position-self._translation,self._rotation.T,axes=((1,),(1,)))

	def reset_position(self,position):
		return self.recenter_position(position) + self._boxcenter
	
	def reset_velocity(self,velocity):
		return scipy.tensordot(self[self.attrs['Velocity']],self._rotation.T,axes=((1,),(1,)))
	
	def distance(self):
		return distance(self.Position)
		
	def flush(self):
		self[self.attrs['Position']] = self.Position
		self[self.attrs['Velocity']] = self.Velocity
		self._rotation = scipy.eye(self._rotation.shape[0],dtype=self._rotation.dtype)
		self._boxcenter[:] = self._translation[:]
	
	@property
	def glos(self):
		return self._translation/scipy.sqrt((self._translation**2).sum(axis=-1))
		
	def cartesian_to_sky(self,wrap=True,degree=True):
		return cartesian_to_sky(self.Position,wrap=wrap,degree=degree)
		
	def apply_rsd(self,velocity_offset,los='local'):

		if not scipy.isscalar(los):
			unit_vector = scipy.array(los,dtype='f8')
			unit_vector /= distance(unit_vector)
		elif los == 'local':
			unit_vector = self.Position/distance(self.Position)[:,None]
		elif los == 'global':
			unit_vector = self.glos
		else:
			axis = los
			if isinstance(los,str): axis = 'xyz'.index(axis)
			unit_vector = scipy.zeros((3),dtype=scipy.float64)
			unit_vector[axis] = 1.
			unit_vector = scipy.tensordot(unit_vector,self._rotation,axes=((0,),(1,)))

		return self.Position + (unit_vector*velocity_offset).sum(axis=-1)[:,None]*unit_vector
		
	def remap(self,u1=(1,0,0),u2=(0,1,0),u3=(0,0,1)):
		cuboid = Cuboid(u1=u1,u2=u2,u3=u3)
		return cuboid.transform((self.Position/self.BoxSize).T).T*self.BoxSize
		
	def subvolume(self,ranges=[[0,1],[0,1],[0,1]]):
		if scipy.isscalar(ranges[0]): ranges = [ranges]*3
		position = self.Position
		mask = self.trues()
		for i,r in enumerate(ranges): mask &= (position[:,i] >= r[0]) & (position[:,i] <= r[1])
		new = self[mask]
		new.BoxSize = scipy.diff(ranges,axis=-1)[:,0]
		new._boxcenter = scipy.array([r[0] for r in ranges],dtype=scipy.float64) + new.BoxSize/2. + self._boxcenter - self._translation
		return new
	
	def apply_operation(self,*operations):
		for operation in operations: getattr(self,operation['method'])(**operation['kwargs'])
	
	def replicate(self,factor=1.1,replicate=[]):
		factors = scipy.zeros((3),dtype='f8')
		factors[:] = factor
		new = self.deepcopy()
		new.BoxSize *= factors
		position = self.attrs['Position']
		if position not in replicate: replicate.append(position)
		shifts = [scipy.arange(-scipy.ceil(factor)+1,scipy.ceil(factor)) for factor in factors]
		columns = {key:[] for key in new}
		for shift in cartesian(shifts):
			tmp = {key: self[key] + self.BoxSize*shift for key in replicate}
			mask = (tmp[position] >= -new.BoxSize/2. + self._boxcenter) & (tmp[position] <= new.BoxSize/2. + self._boxcenter)
			mask = scipy.all(mask,axis=-1)
			for key in new:
				if key in replicate: columns[key].append(tmp[key][mask])
				else: columns[key].append(self[key][mask])
		for key in new:
			new[key] = scipy.concatenate(columns[key],axis=0)
		new._compute_position = True
		new._compute_velocity = True
		return new

class RandomCatalogue(SurveyCatalogue):

	logger = logging.getLogger('RandomCatalogue')
	
	def __init__(self,BoxSize=1.,BoxCenter=0.,size=None,nbar=None,Position='Position',rng=None,seed=None,**attrs):

		self.BoxSize = scipy.empty((3),dtype=scipy.float64)
		self.BoxSize[:] = BoxSize
		self._boxcenter = scipy.empty((3),dtype=scipy.float64)
		self._boxcenter[:] = BoxCenter
		if rng is None: rng = scipy.random.RandomState(seed=seed)
		self.rng = rng
		if size is None: size = rng.poisson(nbar*scipy.prod(self.BoxSize))
		position = scipy.array([rng.uniform(-self.BoxSize[i]/2.+self._boxcenter[i],self.BoxSize[i]/2.+self._boxcenter[i],size=size) for i in range(3)]).T
		super(RandomCatalogue,self).__init__(columns={Position:position},BoxSize=BoxSize,BoxCenter=BoxCenter,Position=Position,seed=seed,size=size,nbar=nbar,**attrs)

class RandomSkyCatalogue(Catalogue):

	logger = logging.getLogger('RandomSkyCatalogue')
	
	def __init__(self,rarange=[0.,360.],decrange=[-90.,90.],nbar=None,rng=None,seed=None,RA='RA',DEC='DEC',wrap=True,**attrs):
		
		area = 4*constants.pi*(rarange[1]-rarange[0])/360./constants.degree**2
		if rng is None: rng = scipy.random.RandomState(seed=seed)
		self.rng = rng
		size = rng.poisson(nbar*area)
		ra = rng.uniform(low=rarange[0],high=rarange[1],size=size)
		dec = scipy.arcsin(1.-rng.uniform(low=0,high=1,size=size)*2.)/constants.degree
		mask = (dec>=decrange[0]) & (dec<=decrange[1])
		ra = ra[mask]
		dec = dec[mask]
		if wrap: ra %= 360.
		super(RandomSkyCatalogue,self).__init__(columns={RA:ra,DEC:dec},seed=seed,size=size,nbar=nbar,**attrs)

def generate_random_redshifts(redshift_density,redshift_to_distance,size=100,factor=3,exact=True,distance_to_redshift=None):

	rng = redshift_density.rng
	zrange = redshift_density.zrange

	drange = redshift_to_distance(zrange)
	distance = rng.uniform(drange[0],drange[1],factor*size)
	if distance_to_redshift is not None:
		redshift = distance_to_redshift(distance)
	else:
		redshift = DistanceToRedshiftArray(distance=comoving_distance,zmax=zrange[1]+1,nz=4096)(distance)
	prob = (distance/drange[1])**2
	assert (prob <= 1.).all()
	mask_redshift = (prob >= rng.uniform(0.,1.,redshift.size)) & redshift_density(redshift)
	nmask_redshift = mask_redshift.sum()
	assert nmask_redshift >= size, 'You should set factor {:.5g}.'.format(factor*len(mask_redshift)*1./nmask_redshift*1.1)
	
	if exact: return redshift[mask_redshift][:size]
	return redshift[mask_redshift]
	
class DistanceToRedshiftArray(object):

	def __init__(self,distance,zmax=100.,nz=2048):
		self.distance = distance
		self.zmax = zmax
		self.nz = nz
		self.prepare()
		
	def prepare(self):
		zgrid = scipy.logspace(-8,scipy.log10(self.zmax),self.nz)
		self.zgrid = scipy.concatenate([[0.], zgrid])
		self.rgrid = self.distance(self.zgrid)
		self.set_interp()

	def set_interp(self):
		self.interp = interpolate.Akima1DInterpolator(self.rgrid,self.zgrid,axis=0)

	def __call__(self,distance):
		return self.interp(distance)


class UniformDensityMask(object):

	logger = logging.getLogger('UniformDensityMask')

	def __init__(self,nbar=1.,rng=None,seed=None):
		self.nbar = nbar
		self.set_rng(rng=rng,seed=seed)

	def set_rng(self,rng=None,seed=None):
		if rng is None: rng = scipy.random.RandomState(seed=seed)
		self.rng = rng

	def prob(self,z,*args,**kwargs):
		return scipy.clip(self.nbar,0.,1.)*scipy.ones(z.shape[-1],dtype='f8')

	def __call__(self,z):
		tmp = self.prob(z)
		return tmp >= self.rng.uniform(low=0.,high=1.,size=len(tmp))

class RedshiftDensityMask(UniformDensityMask):
	
	logger = logging.getLogger('RedshiftDensityMask')
	
	def __init__(self,z=None,nbar=None,zrange=None,path=None,norm=None,rng=None,seed=None,**kwargs):
		if path is not None:
			self.logger.info('Loading density file: {}.'.format(path))
			self.z,self.nbar = scipy.loadtxt(path,unpack=True,**kwargs)
		else:
			self.z,self.nbar = z,nbar
		assert (self.nbar>=0.).all()
		self.zrange = zrange
		zmin,zmax = self.z.min(),self.z.max()
		if self.zrange is None: self.zrange = zmin,zmax
		if not (zmin<=self.zrange[0]) & (zmax>=self.zrange[1]):
			raise ValueError('Redshift range is {:.2f} - {:.2f} when you ask for {:.2f} - {:.2f}.'.format(zmin,zmax,self.zrange[0],self.zrange[1]))
		self.set_rng(rng=rng,seed=seed)
		self.prepare(norm=norm)

	def prepare(self,norm=None):
		if norm is None: norm = 1./self.nbar[self.zmask].max(axis=0)
		self.norm = norm
		self.set_interp()
		return self.norm

	@property
	def zmask(self):
		return (self.z >= self.zrange[0]) & (self.z <= self.zrange[-1])

	def set_interp(self):
		prob = scipy.clip(self.norm*self.nbar,0.,1.)
		self.interp = interpolate.Akima1DInterpolator(self.z,prob,axis=0)

	def prob(self,z):
		toret = self.interp(z)
		mask = (z >= self.zrange[0]) & (z <= self.zrange[-1])
		toret[~mask] = 0.
		return toret
	
	def flatten(self,norm=None):
		if norm is None: norm = self.nbar[self.zmask].sum()
		self.nbar = scipy.ones_like(self.nbar)
		self.prepare(norm=self.zmask.sum()/norm)
	
	def integral(self,z=None,w=None,npoints=1000,normalize=True):
		if z is None and w is None:
			return self.interp.integrate(self.zrange[0],self.zrange[1])
		z,w = self._get_zw_(z=z,w=w,npoints=npoints,normalize=normalize)
		return self._integral_(z,w)

	def _get_zw_(self,z=None,w=None,npoints=1000,normalize=True):
		if z is None:
			z = scipy.linspace(self.zrange[0],self.zrange[1],npoints)
		if w is None:
			w = 1.
			if normalize: w = w/len(z)
			return z,w
		if not isinstance(w,(scipy.ndarray,list)):
			w = w(z)
		if normalize: w = w/w.sum()
		return z,w

	def _integral_(self,z,w):
		return scipy.sum(self.prob(z)*w)

	def normalize(self,factor=1.,**kwargs):

		assert factor <= 1.
		z,w = self._get_zw_(normalize=True,**kwargs)

		def normalization(norm):
			self.norm = norm
			self.set_interp()
			return self._integral_(z,w)/factor - 1.

		min_ = self.nbar[self.nbar>0.].min()
		norm = optimize.brentq(normalization,0.,1/min_) # the lowest point of n(z) is limited by 1.
		
		self.prepare(norm=norm)
		self.logger.info('Norm is: {:.12g}.'.format(self.norm))
		self.logger.info('Expected error: {:.5g}.'.format(self._integral_(z,w)-factor))
		
		return norm
	
	def convert_to_cosmo_edges(self,distance_self,distance_target,zedges=None):
		if zedges is None:
			zedges = (self.z[:-1] + self.z[1:])/2.
			zedges = scipy.concatenate([self.z[0]],zedges,[self.z[-1]])
		dedges = distance_self(zedges)
		volume_self = dedges[1:]**3-dedges[:-1]**3
		dedges = distance_target(zedges)
		volume_target = dedges[1:]**3-dedges[:-1]**3
		self.nbar = self.nbar*volume_self/volume_target
		self.prepare()
	
	def convert_to_cosmo(self,cosmo_self,cosmo_target):
		volume_self = cosmo_self.comoving_distance(self.z)**2/cosmo_self.efunc(self.z)
		volume_target = cosmo_target.comoving_distance(self.z)**2/cosmo_target.efunc(self.z)
		self.nbar = self.nbar*volume_self/volume_target
		self.prepare()

	def __call__(self,z):
		tmp = self.prob(z)
		return tmp >= self.rng.uniform(low=0.,high=1.,size=len(tmp))

class RedshiftDensityMask2D(RedshiftDensityMask):
	
	logger = logging.getLogger('RedshiftDensityMask2D')
	
	def __init__(self,z=None,other=None,nbar=None,zrange=None,norm=None,rng=None,seed=None):
		self.z,self.other,self.nbar = z,other,nbar
		assert (self.nbar>=0.).all()
		self.zrange = zrange
		zmin,zmax = self.z.min(),self.z.max()
		if self.zrange is None: self.zrange = zmin,zmax
		if not (zmin<=self.zrange[0]) & (zmax>=self.zrange[1]):
			raise ValueError('Redshift range in {} is {:.2f} - {:.2f} when you ask for {:.2f} - {:.2f}.'.format(path,zmin,zmax,self.zrange[0],self.zrange[1]))
		self.set_rng(rng=rng,seed=seed)
		self.prepare(norm=norm)

	def set_interp(self):
		prob = scipy.clip(self.norm*self.nbar,0.,1.)
		self.interp = interpolate.RectBivariateSpline(self.z,self.other,prob,bbox=[None,None,None,None],kx=3,ky=3,s=0)

	def prob(self,z,grid=False):
		z,other = z
		toret = self.interp(z,other,grid=grid)
		mask = (z >= self.zrange[0]) & (z <= self.zrange[-1])
		toret[~mask] = 0.
		return toret

	def _get_zw_(self,z=None,w=None,npoints=1000,onpoints=10,orange=[0.,1.],grid=False,normalize=True):
		if z is None:
			z = [scipy.linspace(self.zrange[0],self.zrange[1],npoints),scipy.linspace(range_other[0],range_other[1],onpoints)]
			grid = True
		if w is None:
			w = 1.
			if normalize:
				if grid: w = w/(len(z[0])+len(z[1]))
				else: w = w/len(z)
			return z,w,grid
		if not isinstance(w,(scipy.ndarray,list)):
			w = w(z)
		if normalize: w = w/w.sum()
		return z,w,grid

	def _integral_(self,z,w,grid=False):
		return scipy.sum(self.prob(z,grid=grid)*w)

	def normalize(self,factor=1.,**kwargs):

		assert factor <= 1.
		z,w,grid = self._get_zw_(normalize=True,**kwargs)

		def normalization(norm):
			self.norm = norm
			self.set_interp()
			return self._integral_(z,w,grid=grid)/factor - 1

		min_ = self.prob[self.prob>0.].min()
		norm = optimize.brentq(normalization,0.,1/min_) # the lowest point of n(z) is limited by 1.
		
		self.prepare(norm=norm)
		self.logger.info('Norm is: {:.12g}.'.format(self.norm))
		self.logger.info('Expected error: {:.5g}.'.format(self._integral_(z,w,grid=grid)-factor))
		
		return norm

class KernelDensityMask(RedshiftDensityMask):

	logger = logging.getLogger('KernelDensityMask')
	
	def __init__(self,position,weight=None,distance=None,zrange=None,norm=1.,rng=None,seed=None,**kwargs):
	
		from sklearn.neighbors import KernelDensity
		self.distance = distance
		self.zrange = zrange
		self.set_rng(rng=rng,seed=seed)
		self.kernel = KernelDensity(**kwargs)
		self.kernel.fit(self.get_position(position),sample_weight=weight)
		self.norm = norm
	
	def get_position(self,position):
		if self.distance is not None:
			z,ra,dec = position
			position = sky_to_cartesian(self.distance(z),ra,dec).T
		return position.T

	def prob(self,position,clip=True):
		if self.distance is not None:
			z,ra,dec = position
		logpdf = self.kernel.score_samples(self.get_position(position))
		toret = self.norm*scipy.exp(logpdf)
		if clip: toret = scipy.clip(toret,0.,1.)
		if self.distance is not None and self.zrange is not None:
			mask = (z >= self.zrange[0]) & (z <= self.zrange[-1])
			toret[~mask] = 0.
		return toret

	def integral(self,position,normalize=True):
		toret = scipy.sum(self.prob(position))
		if normalize: toret /= position.shape[-1]
	
	def normalize(self,position,factor=1.):

		assert factor <= 1.
		prob = self.prob(position,clip=False)

		def normalization(norm):
			return scipy.sum(scipy.clip(norm*prob,0.,1.))/(size*factor) - 1.

		min_ = prob[prob>0.].min()
		self.norm = optimize.brentq(normalization,0.,1/min_) # the lowest point of n(z) is limited by 1.
		
		self.logger.info('Norm is: {:.12g}.'.format(self.norm))
		self.logger.info('Expected error: {:.5g}.'.format(self.integral(position,normalize=True)-factor))
		
		return self.norm

class MeshDensityMask(RedshiftDensityMask):

	logger = logging.getLogger('MeshDensityMask')

	def __init__(self,mesh,distance=None,zrange=None,norm=None,rng=None,seed=None,BoxSize=None,BoxCenter=None,dlos=None,mlos=[0.,0.,1.],**kwargs):
		
		self.distance = distance
		self.zrange = zrange
		self.set_rng(rng=rng,seed=seed)
		self.define_rotation_matrix_from_vectors(dlos=dlos,mlos=mlos)
		BoxSize = self.define_cartesian_box_from_boxsize(BoxSize,BoxCenter)
		from nbodykit.lab import ArrayMesh
		self.mesh = ArrayMesh(mesh,BoxSize=BoxSize).to_real_field()
		self.prepare(norm=norm)

	@classmethod
	def from_nbodykit(cls,mesh,distance=None,zrange=None,norm=None,rng=None,seed=None,**kwargs):
		kwargs_ = {}
		kwargs_.update(mesh.attrs)
		kwargs_.update(kwargs)
		return cls(mesh.to_real_field().value,distance=distance,zrange=zrange,norm=norm,rng=rng,seed=seed,**kwargs_)

	def recenter(self,position,weight=None,mlos=[0.,0.,1.],BoxPad=0.):
		position = self.cartesian_transform(position)
		#print self.rotation
		self.define_rotation_matrix_from_position(position,weight=weight,mlos=mlos)
		self.offset = 0.; position = self.affine_transform(position)
		pos_min, pos_max = position.min(axis=0),position.max(axis=0)
		delta = abs(pos_max - pos_min)
		BoxCenter = 0.5 * (pos_min + pos_max)
		delta *= 1.0 + BoxPad
		if (self.BoxSize < delta).any(): raise ValueError('BoxSize too small to contain all data.')
		self.define_cartesian_box_from_boxsize(self.BoxSize,BoxCenter)
		self.prepare(norm=self.norm)

	def prepare(self,norm=None):
		if norm is None: norm = 1./self.max()
		self.norm = norm
		self.set_interp()
		return norm

	@property
	def Nmesh(self):
		return self.mesh.Nmesh

	@property
	def BoxSize(self):
		return self.mesh.BoxSize

	def xgrid(self):
		return [scipy.linspace(0.,boxsize,nmesh) for nmesh,boxsize in zip(self.Nmesh,self.BoxSize)]

	def max(self):
		return max(self.mesh.pm.comm.allgather(self.mesh.value.max()))
	
	def sum(self):
		return self.mesh.csum()

	def std(self):
		return scipy.sqrt((self.mesh**2).cmean()-self.mesh.cmean()**2)

	def abssum(self):
		return abs(self.sum())

	def define_rotation_matrix_from_vectors(self,dlos,mlos=[0.,0.,1.]):
		self.rotation = rotation_matrix_from_vectors(dlos,mlos)

	def define_cartesian_box_from_boxsize(self,BoxSize,BoxCenter):
		BoxSize_ = scipy.empty(3,dtype='f8')
		BoxSize_[:] = BoxSize
		BoxCenter_ = scipy.empty(3,dtype='f8')
		BoxCenter_[:] = BoxCenter
		self.offset = BoxCenter_ - BoxSize_/2.
		return BoxSize_

	def define_rotation_matrix_from_position(self,position,weight=None,mlos=[0.,0.,1.]):
		dlos = scipy.average(position,weights=weight,axis=0)
		self.define_rotation_matrix_from_vectors(dlos=dlos,mlos=mlos)

	def filter_gaussian(self,cov=1.,offset=0.):
		covariance = scipy.zeros(3,dtype='f8')
		covariance[:] = cov
		def filter(k,v):
			return v*scipy.exp(sum(-ki**2/2./cov for ki,cov in zip(k,covariance)))
		self.filter_generic(filter,offset=offset)

	def filter_generic(self,filter,offset=0.):
		mesh = (self.mesh+offset).r2c(out=Ellipsis).apply(filter,kind='wavenumber')
		self.mesh = mesh.c2r(out=Ellipsis)-offset
		self.set_interp()

	def affine_transform(self,position):
		return scipy.tensordot(position,self.rotation,axes=((1,),(1,))) - self.offset

	def cartesian_transform(self,position):
		if self.distance is not None:
			z,ra,dec = position
			return sky_to_cartesian(self.distance(z),ra,dec)
		else:
			return position.T

	def get_position(self,position):
		return self.affine_transform(self.cartesian_transform(position))

	def set_interp(self):
		self.interp = interpolate.RegularGridInterpolator(self.xgrid(),self.mesh,bounds_error=True,fill_value=0.)

	def prob(self,position,clip=True):
		if self.distance is not None:
			z,ra,dec = position
		toret = scipy.zeros(len(z),dtype='f8')
		if self.distance is not None and self.zrange is not None:
			mask = (z >= self.zrange[0]) & (z <= self.zrange[-1])
		else:
			mask = scipy.ones_like(toret,dtype=scipy.bool_)
		#position_ = self.cartesian_transform(position)[mask]
		#print position_.min(axis=0), position_.max(axis=0) 
		toret[mask] = self.norm*self.interp(self.get_position(position)[mask])
		if clip: toret = scipy.clip(toret,0.,1.)
		return toret

	def integral(self,position,normalize=True):
		toret = scipy.sum(self.prob(position))
		if normalize: toret /= position.shape[-1]
	
	def normalize(self,position,factor=1.):

		assert factor <= 1.
		prob = self.prob(position,clip=False)

		def normalization(norm):
			return scipy.sum(scipy.clip(norm*prob,0.,1.))/(size*factor) - 1.

		min_ = prob[prob>0.].min()
		self.norm = optimize.brentq(normalization,0.,1/min_) # the lowest point of n(z) is limited by 1.
		
		self.logger.info('Norm is: {:.12g}.'.format(self.norm))
		self.logger.info('Expected error: {:.5g}.'.format(self.integral(position,normalize=True)-factor))
		
		return self.norm

	def show(self,which=None,axes=[0,1],**kwargs):
		from matplotlib import pyplot
		pyplot.imshow(self.mesh.preview(axes=axes,**kwargs).T,origin='lower',extent=(0,self.BoxSize[axes[0]],0,self.BoxSize[axes[1]]))
		pyplot.colorbar()
		pyplot.show()

	def deepcopy(self):
		new = object.__new__(self.__class__)
		new.__dict__.update(self.__dict__)
		new.offset = self.offset.copy()
		new.mesh = self.mesh.copy()
		new.set_interp()
		return new

	def clip(self,amin,amax):
		self.mesh[self.mesh<amin] = amin
		self.mesh[self.mesh>amax] = amax

	def __mul__(self,other):
		new = self.deepcopy()
		if not isinstance(other,self.__class__):
			new.mesh *= other			
			return new
		assert scipy.all(self.Nmesh == other.Nmesh)
		new.mesh = new.mesh*other.mesh
		new.set_interp()
		return new

	def __div__(self,other):
		new = self.deepcopy()
		if not isinstance(other,self.__class__):
			new.mesh /= other			
			return new
		assert scipy.all(self.Nmesh == other.Nmesh)
		new.mesh = self.mesh/other.mesh
		new.mesh[other.mesh==0.] = self.sum()/other.sum()
		new.set_interp()
		return new
	
	def __add__(self,other):
		new = self.deepcopy()
		if not isinstance(other,self.__class__):
			new.mesh += other			
			return new
		assert scipy.all(self.Nmesh == other.Nmesh)
		new.mesh = self.mesh + other.mesh
		new.set_interp()
		return new

	def __neg__(self):
		new = self.deepcopy()
		new.mesh *= -1
		return new

	def __sub__(self,other):
		return self.__add__(other.__neg__())


class CatalogueMeshDensityMask(MeshDensityMask):

	logger = logging.getLogger('CatalogueMeshDensityMask')

	def __init__(self,position,weight=None,distance=None,zrange=None,norm=None,rng=None,seed=None,Nmesh=256,BoxSize=None,CellSize=None,BoxCenter=None,BoxPad=0.02,resampler='tsc',**kwargs):
		
		self.distance = distance
		self.zrange = zrange
		self.set_rng(rng=rng,seed=seed)
		position = self.cartesian_transform(position)
		self.define_rotation_matrix_from_position(position,weight=weight)
		self.offset = 0.; position = self.affine_transform(position)
		Nmesh,BoxSize = self.define_cartesian_box_from_position(position,Nmesh=Nmesh,BoxSize=BoxSize,CellSize=CellSize,BoxCenter=BoxCenter,BoxPad=BoxPad)
		catalog = self._catalogue_to_nbodykit_(position-self.offset,weight=weight)
		self.mesh = catalog.to_mesh(Nmesh=Nmesh,BoxSize=BoxSize,resampler=resampler).to_real_field(normalize=False)
		self.prepare(norm=norm)
	
	def _catalogue_to_nbodykit_(self,position,weight=None):
		from nbodykit.lab import ArrayCatalog
		dict_ = {'Position':position}
		if weight is not None: dict_.update(Weight=weight)
		return ArrayCatalog({'Position':position,'Weight':weight})

	def define_cartesian_box_from_position(self,position,Nmesh=256,BoxSize=None,CellSize=None,BoxCenter=None,BoxPad=0.02):
		pos_min, pos_max = position.min(axis=0),position.max(axis=0)
		delta = abs(pos_max - pos_min)
		if BoxCenter is None: BoxCenter = 0.5 * (pos_min + pos_max)
		if BoxSize is None:
			BoxSize = delta * (1.0 + BoxPad)
		if (BoxSize < delta).any(): raise ValueError('BoxSize too small to contain all data.')
		if CellSize is not None:
			Nmesh = scipy.ceil(BoxSize/CellSize).astype(int) + 1
		BoxSize = self.define_cartesian_box_from_boxsize(BoxSize,BoxCenter)
		return Nmesh,BoxSize

	def clone(self,position,weight=None,resampler='tsc',norm=None):
		new = object.__new__(self.__class__)
		new.__dict__.update(self.__dict__)
		new.offset = self.offset.copy()
		catalog = self._catalogue_to_nbodykit_(self.get_position(position),weight=weight)
		new.mesh = catalog.to_mesh(Nmesh=self.Nmesh,BoxSize=self.BoxSize,resampler=resampler).to_real_field(normalize=False)
		new.set_interp()
		new.prepare(norm=norm)
		return new

	@classmethod
	def from_mesh(cls,mesh):
		new = object.__new__(cls)
		new.__dict__.update(mesh.__dict__)
		new.offset = new.offset.copy()
		return new

class DensityMaskChunk(object):

	def __init__(self):
		self.density_mask = collections.OrderedDict()
		
	def __iter__(self):
		return self.density_mask.__iter__()
		
	def set_rng(self,rng=None,seed=None):
		for chunkz,density in self.items():
			density.set_rng(rng=rng,seed=seed)

	def keys(self):
		return self.density_mask.keys()
	
	def values(self):
		return self.density_mask.values()
	
	def items(self):
		return self.density_mask.items()
		
	def __getitem__(self,name):
		return self.density_mask[name]
	
	def __setitem__(self,name,item):
		self.density_mask[name] = item
		
	def __delitem__(self,name):
		del self.density_mask[name]
		
	def __call__(self,z,other):
		z = scipy.asarray(z)
		other = scipy.asarray(other)
		toret = scipy.ones(z.shape[-1],dtype=scipy.bool_)
		for ichunkz,density in self.items():
			mask = other == ichunkz
			if mask.any():
				toret[...,mask] = density(z[...,mask])
		return toret

	def __radd__(self,other):
	
		if other == 0: return self
		else: return self.__add__(other)

	def __add__(self,other):

		new = self.__class__()
		new.density_mask = {}
		for ichunkz,density in self.items():
			new[ichunkz] = density
		for ichunkz,density in other.items():
			new[ichunkz] = density
		return new

class GeometryMask(object):

	logger = logging.getLogger('GeometryMask')

	def __init__(self,mask=None,path=None,rng=None,seed=None):
		self.mask = mask
		if path is not None:
			import pymangle
			self.logger.info('Loading geometry file: {}.'.format(path))
			self.mask = pymangle.Mangle(path)
		self.rng = rng
		if rng is None: self.rng = scipy.random.RandomState(seed=seed)
		
	def __call__(self,ra,dec,downsample=True,return_weight=False):
		ids,weight = self.mask.polyid_and_weight(ra,dec)
		mask = ids != -1
		if downsample: mask &= weight >= self.rng.uniform(low=0.,high=1.,size=len(ra))
		if return_weight: return mask,weight
		return mask
