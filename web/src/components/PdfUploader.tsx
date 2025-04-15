"use client";

import { useState } from "react";
import { FileUpload } from "@/components/ui/file-upload";
import { Button } from "@/components/ui/button";
import { generatePresignedUrl } from "@/app/analysis/action";
import { generateReport } from "@/app/analysis/generateReport";
import {
  AlertCircle,
  CheckCircle2,
  Upload,
  FileText,
  AlertTriangle,
} from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";

export function PdfUploader() {
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<{
    success?: boolean;
    message?: string;
  }>({});
  const [viewUrl, setViewUrl] = useState<string | null>(null);
  const [reportData, setReportData] = useState<
    | {
        clause: string;
        risk_category: "Risk" | "No Risk";
        risk_probability: number;
      }[]
    | null
  >(null);
  const [reportError, setReportError] = useState<string | null>(null);

  const handleFileChange = (newFiles: File[]) => {
    // Reset states when file selection changes
    setUploadStatus({});
    setViewUrl(null);
    setReportData(null);
    setReportError(null);

    if (newFiles.length === 0) {
      setFile(null);
      return;
    }

    // Check if it's a PDF file
    const selectedFile = newFiles[0];
    if (selectedFile.type !== "application/pdf") {
      setFile(null);
      setUploadStatus({
        success: false,
        message: "Only PDF files are allowed",
      });
      return;
    }

    setFile(selectedFile);
  };

  const handleUpload = async () => {
    if (!file) {
      setUploadStatus({
        success: false,
        message: "Please select a PDF file first",
      });
      return;
    }

    setUploading(true);
    setUploadStatus({});
    setReportData(null);
    setReportError(null);

    try {
      // Get pre-signed URL from server action
      const { uploadUrl, viewUrl, s3Uri } = await generatePresignedUrl(
        file.name
      );

      // Upload file to pre-signed URL
      const uploadResponse = await fetch(uploadUrl, {
        method: "PUT",
        body: file,
        headers: {
          "Content-Type": file.type,
        },
      });

      if (!uploadResponse.ok) {
        throw new Error("Failed to upload file");
      }

      setUploadStatus({
        success: true,
        message: "File uploaded successfully!",
      });

      setViewUrl(viewUrl);

      // Generate report
      setGenerating(true);
      const res = await generateReport(s3Uri);

      if (res.success && res.data) {
        // Sort data by risk probability (highest first)
        const sortedData = [...res.data].sort(
          (a, b) => b.risk_probability - a.risk_probability
        );
        setReportData(sortedData);
      } else {
        setReportError(res.message || "Failed to generate report");
      }
    } catch (error) {
      console.error("Upload error:", error);
      setUploadStatus({
        success: false,
        message: "Failed to upload file. Please try again.",
      });
    } finally {
      setUploading(false);
      setGenerating(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="border-2 border-dashed border-gray-300 dark:border-gray-700 rounded-lg p-6">
        {file ? (
          <div className="flex items-center justify-between bg-gray-900 dark:bg-gray-800 p-4 rounded">
            <div className="flex items-center gap-3">
              <div className="bg-gray-800 dark:bg-gray-700 p-2 rounded">
                <FileText className="h-5 w-5 text-blue-100" />
              </div>
              <div>
                <p className="font-medium text-white">{file.name}</p>
                <p className="text-sm text-gray-400">
                  {(file.size / 1024 / 1024).toFixed(2)} MB
                </p>
              </div>
            </div>
            <Button
              variant="outline"
              onClick={() => {
                setFile(null);
                setViewUrl(null);
                setUploadStatus({});
                setReportData(null);
                setReportError(null);
              }}
              disabled={uploading || generating}
            >
              Remove
            </Button>
          </div>
        ) : (
          <FileUpload onChange={handleFileChange} />
        )}
      </div>

      <div className="flex justify-center">
        <Button
          onClick={handleUpload}
          disabled={!file || uploading || generating}
          className="w-full max-w-xs"
        >
          {generating ? (
            <>
              <span className="mr-2 h-4 w-4 animate-spin inline-block border-2 border-current border-t-transparent rounded-full" />
              Generating Report...
            </>
          ) : uploading ? (
            <>
              <Upload className="mr-2 h-4 w-4 animate-pulse" />
              Uploading...
            </>
          ) : (
            "Upload & Analyze PDF"
          )}
        </Button>
      </div>

      {uploadStatus.message && (
        <Alert variant={uploadStatus.success ? "default" : "destructive"}>
          {uploadStatus.success ? (
            <CheckCircle2 className="h-4 w-4" />
          ) : (
            <AlertCircle className="h-4 w-4" />
          )}
          <AlertTitle>{uploadStatus.success ? "Success" : "Error"}</AlertTitle>
          <AlertDescription>{uploadStatus.message}</AlertDescription>
        </Alert>
      )}

      {/* Display content in sequence - PDF first, then report */}
      <div className="flex flex-col gap-6">
        {viewUrl && (
          <div className="border rounded-lg p-4 bg-gray-50 dark:bg-gray-900 dark:border-gray-800">
            <h2 className="text-xl font-semibold mb-4 flex items-center">
              <FileText className="mr-2 h-5 w-5" />
              Uploaded PDF
            </h2>
            <div className="border dark:border-gray-700 rounded-lg overflow-hidden h-96">
              <iframe
                src={viewUrl}
                className="w-full h-full bg-white"
                title="Uploaded PDF"
              />
            </div>
          </div>
        )}

        {generating && !reportData && !reportError && (
          <div className="border dark:border-gray-800 rounded-lg p-6 flex flex-col items-center justify-center h-96 bg-gray-50 dark:bg-gray-900">
            <span className="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mb-4"></span>
            <p className="text-lg font-medium">Analyzing document...</p>
            <p className="text-gray-500 dark:text-gray-400 text-sm mt-2">
              This may take a few moments
            </p>
          </div>
        )}

        {reportError && (
          <div className="border border-red-200 dark:border-red-900 rounded-lg p-6 bg-red-50 dark:bg-red-950/30">
            <div className="flex items-center mb-4">
              <AlertCircle className="h-5 w-5 text-red-500 mr-2" />
              <h2 className="text-xl font-semibold text-red-700 dark:text-red-400">
                Report Generation Failed
              </h2>
            </div>
            <p className="text-red-600 dark:text-red-400">{reportError}</p>
          </div>
        )}

        {reportData && reportData.length > 0 && (
          <div className="border rounded-lg p-4 bg-gray-50 dark:bg-gray-900 dark:border-gray-800">
            <h2 className="text-xl font-semibold mb-4 flex items-center">
              <AlertCircle className="mr-2 h-5 w-5" />
              Risk Analysis Report
            </h2>
            <div className="overflow-auto max-h-96 pr-2">
              {reportData.map((item, index) => (
                <div
                  key={index}
                  className={`mb-4 p-4 rounded-lg border ${
                    item.risk_category === "Risk"
                      ? "border-red-200 bg-red-50 dark:border-red-900 dark:bg-red-950/30"
                      : "border-green-200 bg-green-50 dark:border-green-900 dark:bg-green-950/30"
                  }`}
                >
                  <div className="flex justify-between items-start mb-2">
                    <div className="flex items-center">
                      {item.risk_category === "Risk" ? (
                        <AlertTriangle className="h-5 w-5 text-red-500 mr-2" />
                      ) : (
                        <CheckCircle2 className="h-5 w-5 text-green-500 mr-2" />
                      )}
                      <h3 className="font-medium">
                        {item.risk_category === "Risk"
                          ? "Risk Detected"
                          : "No Risk"}
                      </h3>
                    </div>
                    <div className="text-sm font-medium px-2 py-1 rounded-full bg-gray-200 dark:bg-gray-700">
                      {(item.risk_probability * 100).toFixed(1)}% probability
                    </div>
                  </div>
                  <p className="text-sm mt-2 pl-7">{item.clause}</p>
                </div>
              ))}

              {reportData.length === 0 && (
                <div className="p-4 text-center">
                  <CheckCircle2 className="h-8 w-8 text-green-500 mx-auto mb-2" />
                  <p>No risks detected in this document.</p>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
