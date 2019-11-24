import scipy
from numpy import testing
from pymakesurvey import *

setup_logging()

def test_remap():

	kwargs = dict(u1=(2,1,0), u2=(0,1,1), u3=(1,0,0))
	position = [scipy.random.uniform(0.,1.,10000) for i in range(3)]
	c = Cuboid(**kwargs)
	test = c.transform(position)
	
	import test_remap
	c = test_remap.Cuboid(**kwargs)
	for i,(x,y,z) in enumerate(zip(position[0],position[1],position[2])):
		#print (test[0][i], test[1][i], test[2][i]), c.Transform(x,y,z)
		testing.assert_allclose((test[0][i],test[1][i],test[2][i]),c.Transform(x,y,z),rtol=1e-7,atol=1e-7)
		
def test_catalogue():

	catalogue = RandomCatalogue(BoxSize=10.,size=1000,BoxCenter=3.,name='randoms')
	
	position = catalogue.Position
	name = catalogue.attrs['name']
	new = catalogue.subvolume([0.,4.])
	testing.assert_allclose(new.BoxSize,[4.,4.,4.],rtol=1e-7,atol=1e-7)
	testing.assert_allclose(new._BoxCenter,[2.,2.,2.],rtol=1e-7,atol=1e-7)
	new.attrs['name'] = 'subrandoms'
	assert catalogue.attrs['name'] == name
	new['Position'] -= 1.
	assert (catalogue.Position == position).all()
	new['Position'] += 1.
	new.recenter()
	assert ((new.Position >= -2.) & (new.Position <= 2.)).all()
	position = new.Position
	new.rotate_about_origin_axis(axis='x',angle=90.)
	testing.assert_allclose(new.Position,position[:,[0,2,1]]*scipy.array([1,-1,1]),rtol=1e-7,atol=1e-7)
	new.reset_rotate_about_origin()
	new.rotate_about_origin_axis(axis='y',angle=90.)
	testing.assert_allclose(new.Position,position[:,[2,1,0]]*scipy.array([1,1,-1]),rtol=1e-7,atol=1e-7)
	new.reset_rotate_about_origin()
	new.rotate_about_origin_axis(axis='z',angle=90.)
	testing.assert_allclose(new.Position,position[:,[1,0,2]]*scipy.array([-1,1,1]),rtol=1e-7,atol=1e-7)
	distance = new.distance()
	new.rotate_about_origin_axis(axis=scipy.random.randint(0,3),angle=scipy.random.uniform(0.,360.))
	testing.assert_allclose(new.distance(),distance,rtol=1e-7,atol=1e-7)
	
def test_cutsky():

	drange = [10.,20.]; rarange = scipy.array([0.,50.])-25.+20.; decrange = [-5.,5.]
	boxsize,operations = cutsky_to_box(drange=drange,rarange=rarange,decrange=decrange)
	deltara,deltadec,dmin = box_to_cutsky(boxsize=boxsize,dmax=drange[-1])
	testing.assert_allclose(dmin,drange[0],rtol=1e-7,atol=1e-7)
	testing.assert_allclose(deltara,abs(rarange[1]-rarange[0]),rtol=1e-7,atol=1e-7)
	testing.assert_allclose(deltadec,abs(decrange[1]-decrange[0]),rtol=1e-7,atol=1e-7)
	catalogue = RandomCatalogue(BoxSize=boxsize,size=100000,BoxCenter=0.,name='randoms')
	catalogue.recenter()
	catalogue.apply_operation(*operations)
	catalogue['distance'],catalogue['RA'],catalogue['DEC'] = catalogue.cartesian_to_sky(wrap=False)
	for field in ['distance','RA','DEC']: print field, catalogue[field].min(), catalogue[field].max()

def test_redshift_array(nz=4096):
	from astropy import cosmology
	cosmo = cosmology.wCDM(H0=0.71,Om0=0.31,Ode0=0.69,w0=-1)
	zmax = 10.
	distance = lambda z: cosmo.comoving_distance(z).value*cosmo.h
	redshift = DistanceToRedshiftArray(distance=distance,zmax=zmax,nz=nz)
	z = scipy.random.uniform(0.,2.,10000)
	print scipy.absolute(redshift(distance(z))-z).max()

def test_density():

	n = 100; zrange = [0.6,1.1]
	z = scipy.linspace(0.5,1.5,n)
	nbar = scipy.ones((n),dtype=scipy.float64)
	density = RedshiftDensityMask(z=z,nbar=nbar,zrange=zrange)
	mask = density(z)
	assert mask[(z>=zrange[0]) & (z<=zrange[0])].all()
	density.normalize(0.5)

def test_rotation_matrix():
	def norm(v):
		return scipy.sqrt(scipy.dot(v,v))
	a = [1819.25599061,340.48034526,2.1809526 ]
	b = [0.,0.,1.]
	rot = rotation_matrix_from_vectors(a,b)
	print rot
	print rot.dot(a/norm(a))
	a = scipy.array([1.,1.,0.])
	b = scipy.array([2.3,12.,2.])
	#b = scipy.array([0.,2.,0.])
	rot = rotation_matrix_from_vectors(a,b)
	print b/norm(b)
	print rot.dot(a/norm(a))
	print rot
	a = scipy.array([1.,1.,0.])
	rot = rotation_matrix_from_vectors(a,a)
	print rot

#test_remap()
#test_catalogue()
#test_cutsky()
#test_density()
#test_rotation_matrix()
test_redshift_array()
