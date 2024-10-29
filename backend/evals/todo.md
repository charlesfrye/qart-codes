# consider
- eval is two steps: 
- 1) batch generate from a bunch of prompts, n=many for each prompt for pass@ reporting
- 2) eval on batch output, report results
  
ideally do all of this in w&b using some run ID for tracking history, add images as artifacts of the run