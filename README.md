# About
Repository for Generalised Zero Shot Learning Model which is being created while working on Bachelor Thesis at Warsaw University of Technology

## Dataset
In this repository dataset is ignored. To use the model properly you have to:
1. Install the [AWS CLI](https://aws.amazon.com/cli/)
2. Run `aws s3 sync --no-sign-request --region <your-region> "s3://cse-cic-ids2018/" dest-dir` [AWS region list](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/using-regions-availability-zones.html#region-name)
3. Put data inside `/data/raw`