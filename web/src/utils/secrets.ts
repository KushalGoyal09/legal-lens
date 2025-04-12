export const getAwsSecret = () => {
  if (process.env.AWS_SECRET_ACCESS_KEY) {
    return process.env.AWS_SECRET_ACCESS_KEY;
  } else {
    throw new Error("AWS_SECRET_ACCESS_KEY is not set");
  }
};

export const getAwsAccessKeyId = () => {
  if (process.env.AWS_ACCESS_KEY_ID) {
    return process.env.AWS_ACCESS_KEY_ID;
  } else {
    throw new Error("AWS_ACCESS_KEY_ID is not set");
  }
};

export const getAwsRegion = () => {
  if (process.env.AWS_REGION) {
    return process.env.AWS_REGION;
  } else {
    throw new Error("AWS_REGION is not set");
  }
};

export const getAwsBucketName = () => {
  if (process.env.AWS_BUCKET_NAME) {
    return process.env.AWS_BUCKET_NAME;
  } else {
    throw new Error("AWS_BUCKET_NAME is not set");
  }
};
