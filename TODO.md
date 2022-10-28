# TODO for integrating gradients with OQuPy

### Broad overview

- [ ] Fix the wires on the MPO for the backpropagation: i know how to do it
- [ ] Decide on the broad structural form of the high level function that the user will use
- [ ] Decide on whether this algorithm should be implemented within a single for loop iteration


- [ ] Write backprop algorithm proper and debug
- [ ] Write high level function that combines dprop and dsys
- [ ] Implement tutorial using ideally autograd and finite difference

- [ ] Reduce the list of currents once they have been contracted to produce dprop_list as they no longer take up memory
