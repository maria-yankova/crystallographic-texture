# Disorientation Algorithm

## Requirements

* For each grain:
    * Euler angles
    * List of neighbouring grains
    * Crystal structure
* Symmetry operations associated with each distinct crystal structure

## Algorithm for finding disorientations between neighbouring grains

1. Start with array of shape `(N, 3)` for `N` grains and three Euler angles per grain (in Bunge notation) to represent the orientation of each grain relative to a sample reference frame.
2. Transform Euler angle triplets into rotation matrices, resulting in an orientation array `R` of shape `(N, 3, 3)`. 
3. Use a neighbour list of shape `(M, 2)` which indexes `M` neighbouring grain pairs, to form arrays of shape `(M, 3, 3)` $R_A$ and $R_B$.
4. For all distinct crystal structures, form unique crystal structure pairs.
    * Order the neighbour pair indices consistently according to their crystal structure indices. (So, a neighbour pair with crystal structure indices `(0, 1)` is the same as that with `(1, 0)`.)
5. For each unique crystal structure pair,
    * form a subset of $R_A$ and $R_B$, whose respective grains belong to that crystal structure pair
    * select symmetry operations associated with each crystal structure
    * apply the symmetry operations to $R_A$ and $R_B$ to form  $R_\textnormal{A,sym}$ and $R_\textnormal{B,sym}$
    * broadcast $R_\textnormal{A,sym}$ and $R_\textnormal{B,sym}$ so that they are the same size
    * find all misorientations between A and B using $R_\textnormal{AB,sym} = R_\textnormal{B,sym}R_{A,sym}^{-1}$


## Scenarios

1. For a set of grains with Euler angles, neighbour list and crystal structures, calculate the grain boundary the disorientation axes and angles for the whole data set (Optionally, make accessible all misorientations).
2. Choose a given crystal structure pair as crystal structure indices
3. Choose a given grain pair as grain indices

