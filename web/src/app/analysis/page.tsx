import { PdfUploader } from "@/components/PdfUploader"

export default function Home() {
  return (
    <main className="container mx-auto py-10 px-4">
      <h1 className="text-3xl font-bold mb-6 text-center">PDF File Uploader</h1>
      <div className="max-w-3xl mx-auto">
        <PdfUploader />
      </div>
    </main>
  )
}

