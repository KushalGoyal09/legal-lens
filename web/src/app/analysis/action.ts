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

export async function generateReport(s3Uri: string) {
  console.log(`Upload complete for file: ${s3Uri}`);
  try {
    const res = await fetch("http://localhost:5000/api/report", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ url: s3Uri }),
    });
    const data: Array<{
      clause: string;
      risk_category: "Risk" | "No Risk";
      risk_probability: number;
    }> = await res.json();

    return {
      success: true,
      message: "Report generated successfully",
      data: data.filter((item) => {
        return item.risk_category === "Risk";
      }),
    };
  } catch (error) {
    console.log(error);
    return {
      success: false,
      message: "Error generating report",
      data: null,
    };
  }
}
