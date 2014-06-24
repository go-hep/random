package random

import (
	"math"
	"math/rand"
)

// Dist is a distribution function
type Dist func() float64

// DiscrDist a discrete distribution function
type DiscrDist func() int64

// Gauss returns a normally distributed float64 in the range
// [-math.MaxFloat64, +math.MaxFloat64] with standard normal distribution of
// mean=mean and stddev=stddev.
func Gauss(mean, stddev float64, src *rand.Source) Dist {
	fct := func() float64 {
		return rand.NormFloat64()*stddev + mean
	}
	if src != nil {
		r := rand.New(*src)
		fct = func() float64 {
			return r.NormFloat64()*stddev + mean
		}
	}
	return Dist(fct)
}

// Exp returns an exponentially distributed float64
func Exp(mean float64, src *rand.Source) Dist {
	fct := func() float64 {
		return rand.ExpFloat64() / mean
	}
	if src != nil {
		r := rand.New(*src)
		fct = func() float64 {
			return r.ExpFloat64() / mean
		}
	}
	return Dist(fct)
}

// Chi2 returns a Chi2 distributed random number generation function
func Chi2(ndf int64, src *rand.Source) Dist {
	norm := Gauss(0, 1, src)
	fct := func() float64 {
		x := 0.
		for i := int64(0); i < ndf; i++ {
			n := norm()
			x += n * n
		}
		return x
	}
	return Dist(fct)
}

// Poisson
func Poisson(mean float64, src *rand.Source) DiscrDist {
	flat := Flat(0, 1, src)
	fct := func() int64 {
		i := int64(0)
		t := math.Exp(-mean)
		p := 1.0
		for ; p > t; p *= flat() {
			i += 1
		}
		return i
	}
	return DiscrDist(fct)
}

// Flat
func Flat(min, max float64, src *rand.Source) Dist {
	delta := max - min
	fct := func() float64 {
		return rand.Float64()*delta + min
	}
	if src != nil {
		r := rand.New(*src)
		fct = func() float64 {
			return r.Float64()*delta + min
		}
	}
	return Dist(fct)
}

// Bernoulli
func Bernoulli(p float64, src *rand.Source) DiscrDist {
	uniform := Flat(0., 1., src)
	fct := func() int64 {
		if uniform() < p {
			return 1
		}
		return 0
	}
	return DiscrDist(fct)
}

// Binomial
func Binomial(n int64, p float64, src *rand.Source) DiscrDist {
	b := Bernoulli(p, src)
	fct := func() int64 {
		x := int64(0)
		for i := int64(0); i <= n; i++ {
			x += b()
		}
		return x
	}
	return DiscrDist(fct)
}

// EOF
