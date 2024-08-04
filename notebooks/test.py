import mitsuba as mi

mi.set_variant("scalar_rgb")

a = mi.quad.gauss_legendre(24)
print(type(a[0]))    
