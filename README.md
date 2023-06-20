# TODOs

## Frontend
- More parameters: "Negative Prompt" and two sliders: "QR code < -- > Art" "Quick & Dirty < -- > Slow & Purdy"
  - awaiting backend changes; this is a @charlesfrye todo
- Improve the default state/user nudging
  - Randomly choose a prompt from a set of nice ones
  - Randomly choose QR text between the short link, the GitHub repo, Rickroll, etc
- "Verify scan" button
  - This is a @charlesfrye todo, part of the MLOps improvements
- Route for prompt + qr, and opengraph images for those routes
  - as part of sharing features

## Backend
- prepare to allow more config params from users
- run a sweep over hyperparameters to pick good defaults
- find a way to change the start/stop time, as in the WebUI
- deliver the results directly from the stub.Dict kv store
  - and move the cloud storage to a separate thread

## Skunkworks
- write a natural language interface
