"use server";

import {
  getAwsAccessKeyId,
  getAwsBucketName,
  getAwsRegion,
  getAwsSecret,
} from "@/utils/secrets";
import { S3Client, PutObjectCommand } from "@aws-sdk/client-s3";
import { getSignedUrl } from "@aws-sdk/s3-request-presigner";
import crypto from "crypto";

const s3Client = new S3Client({
  region: getAwsRegion(),
  credentials: {
    accessKeyId: getAwsAccessKeyId(),
    secretAccessKey: getAwsSecret(),
  },
});

const BUCKET_NAME = getAwsBucketName();

export async function generatePresignedUrl(fileName: string) {
  const randomId = crypto.randomBytes(16).toString("hex");
  const fileKey = `documents/${randomId}-${fileName}`;
  const command = new PutObjectCommand({
    Bucket: BUCKET_NAME,
    Key: fileKey,
    ContentType: "application/pdf",
  });
  const uploadUrl = await getSignedUrl(s3Client, command, { expiresIn: 3600 });
  const s3Uri = `s3://${BUCKET_NAME}/${fileKey}`;
  const viewUrl = `https://s3.${getAwsRegion()}.amazonaws.com/${BUCKET_NAME}/${fileKey}`;

  return { uploadUrl, viewUrl, s3Uri };
}
