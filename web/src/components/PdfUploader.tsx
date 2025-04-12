"use client"

import { useState } from "react"
import { FileUpload } from "@/components/ui/file-upload"
import { Button } from "@/components/ui/button"
import { generatePresignedUrl, uploadComplete } from "@/app/analysis/action"
import { AlertCircle, CheckCircle2 } from "lucide-react"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"

export function PdfUploader() {
  const [files, setFiles] = useState<File[]>([])
  const [uploading, setUploading] = useState(false)
  const [uploadStatus, setUploadStatus] = useState<{
    success?: boolean
    message?: string
  }>({})
  const [viewUrl, setViewUrl] = useState<string | null>(null)

  const handleFileChange = (newFiles: File[]) => {
    // Only keep PDF files
    const pdfFiles = newFiles.filter((file) => file.type === "application/pdf")

    if (pdfFiles.length === 0 && newFiles.length > 0) {
      setUploadStatus({
        success: false,
        message: "Only PDF files are allowed",
      })
      return
    }

    // Only keep the most recent file
    setFiles([pdfFiles[0]])
    setUploadStatus({})
    setViewUrl(null)
  }

  const handleUpload = async () => {
    if (!files.length) {
      setUploadStatus({
        success: false,
        message: "Please select a PDF file first",
      })
      return
    }

    const file = files[0]
    setUploading(true)
    setUploadStatus({})

    try {
      // Get pre-signed URL from server action
      const { uploadUrl, fileKey, viewUrl } = await generatePresignedUrl(file.name)

      // Upload file to pre-signed URL
      const uploadResponse = await fetch(uploadUrl, {
        method: "PUT",
        body: file,
        headers: {
          "Content-Type": file.type,
        },
      })

      if (!uploadResponse.ok) {
        throw new Error("Failed to upload file")
      }

      // Notify server that upload is complete
      await uploadComplete(fileKey)

      setUploadStatus({
        success: true,
        message: "File uploaded successfully!",
      })

      setViewUrl(viewUrl)
    } catch (error) {
      console.error("Upload error:", error)
      setUploadStatus({
        success: false,
        message: "Failed to upload file. Please try again.",
      })
    } finally {
      setUploading(false)
    }
  }

  return (
    <div className="space-y-6">
      <FileUpload onChange={handleFileChange} />

      <div className="flex justify-center">
        <Button onClick={handleUpload} disabled={!files.length || uploading} className="w-full max-w-xs">
          {uploading ? "Uploading..." : "Upload PDF"}
        </Button>
      </div>

      {uploadStatus.message && (
        <Alert variant={uploadStatus.success ? "default" : "destructive"}>
          {uploadStatus.success ? <CheckCircle2 className="h-4 w-4" /> : <AlertCircle className="h-4 w-4" />}
          <AlertTitle>{uploadStatus.success ? "Success" : "Error"}</AlertTitle>
          <AlertDescription>{uploadStatus.message}</AlertDescription>
        </Alert>
      )}

      {viewUrl && (
        <div className="mt-8">
          <h2 className="text-xl font-semibold mb-4">Uploaded PDF</h2>
          <div className="border rounded-lg overflow-hidden h-[500px]">
            <iframe src={viewUrl} className="w-full h-full" title="Uploaded PDF" />
          </div>
        </div>
      )}
    </div>
  )
}

