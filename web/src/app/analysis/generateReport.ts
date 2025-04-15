export async function generateReport(s3Uri: string) {
  const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000";

  try {
    const res = await fetch(`${apiUrl}api/report`, {
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
