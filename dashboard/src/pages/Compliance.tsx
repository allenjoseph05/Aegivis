import { useParams, Link } from "react-router-dom";
import { ArrowLeft, FileText } from "lucide-react";
import { ReportForm } from "../components/ReportForm";

export function Compliance() {
  const { sessionId } = useParams<{ sessionId: string }>();

  if (!sessionId) return null;

  return (
    <div className="p-6 max-w-4xl mx-auto space-y-6">
      {/* Back */}
      <Link
        to={`/sessions/${sessionId}`}
        className="inline-flex items-center gap-1 text-sm text-gray-500 hover:text-gray-700"
      >
        <ArrowLeft size={14} />
        Back to Session
      </Link>

      {/* Header */}
      <div className="bg-white rounded-xl shadow-sm border p-5">
        <div className="flex items-center gap-3 mb-1">
          <FileText size={20} className="text-blue-600" />
          <h1 className="text-lg font-bold text-gray-900">Compliance Reports</h1>
        </div>
        <p className="text-sm text-gray-500">
          Generate structured compliance evidence packages for regulatory requirements.
        </p>
        <div className="font-mono text-xs text-gray-400 mt-2">Session: {sessionId}</div>
      </div>

      {/* Report form */}
      <div className="bg-white rounded-xl shadow-sm border p-5">
        <ReportForm sessionId={sessionId} />
      </div>
    </div>
  );
}
