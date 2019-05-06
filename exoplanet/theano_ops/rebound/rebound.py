#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["ReboundOp"]

import numpy as np

from theano import gof
import theano.tensor as tt


class ReboundOp(gof.Op):

    __props__ = ()

    def make_node(self, masses, coords, times):
        in_args = [
            tt.as_tensor_variable(masses),
            tt.as_tensor_variable(coords),
            tt.as_tensor_variable(times)]
        out_args = [
            (tt.shape_padright(in_args[2])+tt.shape_padleft(in_args[1])).type(),
        ]
        return gof.Apply(self, in_args, out_args)

    # def infer_shape(self, node, shapes):
    #     return tt.concatenate((shapes[1], shapes[2])),

    def perform(self, node, inputs, outputs):
        # NOTE: Units should be AU, M_sun, year/2pi
        import rebound

        masses, coords, times = inputs

        num = len(masses)
        results = np.empty(list(times.shape) + list(coords.shape))

        sim = rebound.Simulation()
        sim.G = masses[0]**2
        sim.add(m=1., x=coords[0, 0], y=coords[0, 1], z=coords[0, 2],
                vx=coords[0, 3], vy=coords[0, 4], vz=coords[0, 5])
        for i in range(1, num):
            sim.add(# primary=sim.particles[0],
                    m=masses[i] / masses[0],
                    x=coords[i, 0], y=coords[i, 1], z=coords[i, 2],
                    vx=coords[i, 3], vy=coords[i, 4], vz=coords[i, 5])

        for i, t in enumerate(times):
            sim.integrate(t - sim.t)
            for j, part in enumerate(sim.particles):
                for k, coord in enumerate("x y z vx vy vz".split()):
                    results[i, j, k] = getattr(part, coord)

        outputs[0][0] = results
