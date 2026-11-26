import { useMemo, useState, type ChangeEvent } from 'react';
import {
  AlertTriangle,
  CheckCircle2,
  Download,
  FileSpreadsheet,
  Play,
  Plus,
  RotateCcw,
  Trash2,
  Upload,
} from 'lucide-react';
import type { CustomerRecord, ManualCustomerInput, SegmentsPayload } from '../types';

interface DataWorkspaceProps {
  segments: SegmentsPayload;
  activeCustomers: CustomerRecord[];
  onPipelineUpdated: (nextSegments: SegmentsPayload) => Promise<void>;
  onNavigateBrief: () => void;
  onNavigateGeometry: () => void;
}

const MANUAL_TEMPLATE_ROWS: ManualCustomerInput[] = [
  { customer_id: 1, gender: 'Female', age: 23, annual_income_k: 21, spending_score: 79 },
  { customer_id: 2, gender: 'Male', age: 25, annual_income_k: 24, spending_score: 83 },
  { customer_id: 3, gender: 'Female', age: 22, annual_income_k: 27, spending_score: 76 },
  { customer_id: 4, gender: 'Male', age: 26, annual_income_k: 29, spending_score: 81 },
  { customer_id: 5, gender: 'Female', age: 24, annual_income_k: 22, spending_score: 74 },
  { customer_id: 6, gender: 'Male', age: 27, annual_income_k: 31, spending_score: 78 },
  { customer_id: 7, gender: 'Female', age: 45, annual_income_k: 22, spending_score: 18 },
  { customer_id: 8, gender: 'Male', age: 49, annual_income_k: 25, spending_score: 15 },
  { customer_id: 9, gender: 'Female', age: 42, annual_income_k: 28, spending_score: 22 },
  { customer_id: 10, gender: 'Male', age: 52, annual_income_k: 26, spending_score: 19 },
  { customer_id: 11, gender: 'Female', age: 47, annual_income_k: 24, spending_score: 25 },
  { customer_id: 12, gender: 'Male', age: 44, annual_income_k: 29, spending_score: 16 },
  { customer_id: 13, gender: 'Female', age: 39, annual_income_k: 54, spending_score: 51 },
  { customer_id: 14, gender: 'Male', age: 43, annual_income_k: 56, spending_score: 48 },
  { customer_id: 15, gender: 'Female', age: 41, annual_income_k: 58, spending_score: 53 },
  { customer_id: 16, gender: 'Male', age: 38, annual_income_k: 52, spending_score: 49 },
  { customer_id: 17, gender: 'Female', age: 46, annual_income_k: 55, spending_score: 46 },
  { customer_id: 18, gender: 'Male', age: 40, annual_income_k: 59, spending_score: 52 },
  { customer_id: 19, gender: 'Female', age: 31, annual_income_k: 88, spending_score: 86 },
  { customer_id: 20, gender: 'Male', age: 34, annual_income_k: 92, spending_score: 84 },
  { customer_id: 21, gender: 'Female', age: 32, annual_income_k: 85, spending_score: 89 },
  { customer_id: 22, gender: 'Male', age: 35, annual_income_k: 96, spending_score: 81 },
  { customer_id: 23, gender: 'Female', age: 30, annual_income_k: 90, spending_score: 83 },
  { customer_id: 24, gender: 'Male', age: 33, annual_income_k: 94, spending_score: 88 },
  { customer_id: 25, gender: 'Male', age: 42, annual_income_k: 88, spending_score: 17 },
  { customer_id: 26, gender: 'Female', age: 40, annual_income_k: 91, spending_score: 21 },
  { customer_id: 27, gender: 'Male', age: 45, annual_income_k: 95, spending_score: 14 },
  { customer_id: 28, gender: 'Female', age: 39, annual_income_k: 86, spending_score: 19 },
  { customer_id: 29, gender: 'Male', age: 43, annual_income_k: 93, spending_score: 23 },
  { customer_id: 30, gender: 'Female', age: 32, annual_income_k: 45, spending_score: 68 },
];

function customerRecordsToInputs(customers: CustomerRecord[]): ManualCustomerInput[] {
  const trainingOnly = customers.filter((c) => c.source === 'training');
  const sourceList = trainingOnly.length >= 15 ? trainingOnly : customers;
  return sourceList.map((c, idx) => {
    const parsedRef = c.customer_ref ? Number(c.customer_ref.replace(/\D+/g, '')) : NaN;
    return {
      customer_id: Number.isFinite(parsedRef) && parsedRef > 0 ? parsedRef : idx + 1,
      gender: c.gender === 'Male' ? 'Male' : 'Female',
      age: Math.round(c.age),
      annual_income_k: Number(c.annual_income_k),
      spending_score: Number(c.spending_score),
    };
  });
}

function parseDelimitedText(rawText: string): ManualCustomerInput[] {
  const lines = rawText
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  if (lines.length < 2) {
    throw new Error('Provide a header row and at least 1 customer row.');
  }

  const delimiter = lines[0].includes('\t') ? '\t' : ',';
  const headers = lines[0].split(delimiter).map((h) => h.trim().toLowerCase());

  const idIdx = headers.findIndex((h) => h === 'customerid' || h === 'customer_id' || h === 'id');
  const gIdx = headers.findIndex((h) => h === 'gender' || h === 'sex');
  const aIdx = headers.findIndex((h) => h === 'age');
  const iIdx = headers.findIndex((h) => h.includes('income'));
  const sIdx = headers.findIndex((h) => h.includes('spending') || h === 'score');

  if (gIdx < 0 || aIdx < 0 || iIdx < 0 || sIdx < 0) {
    throw new Error(
      'Required columns missing: include Gender, Age, Annual Income (k$), and Spending Score (1-100).'
    );
  }

  const parsed: ManualCustomerInput[] = [];
  const usedIds = new Set<number>();

  for (let r = 1; r < Math.min(lines.length, 5001); r += 1) {
    const cols = lines[r].split(delimiter).map((c) => c.trim());
    if (cols.length < 4) {
      throw new Error(`Row ${r + 1} has fewer than 4 columns.`);
    }

    const rawGender = cols[gIdx].toLowerCase();
    const gender: 'Male' | 'Female' =
      rawGender === 'male' || rawGender === 'm' ? 'Male' : 'Female';
    const age = Number(cols[aIdx]);
    const income = Number(cols[iIdx]);
    const spending = Number(cols[sIdx]);

    if (!Number.isFinite(age) || age < 16 || age > 100) {
      throw new Error(`Row ${r + 1}: Age must be between 16 and 100.`);
    }
    if (!Number.isFinite(income) || income < 5 || income > 500) {
      throw new Error(`Row ${r + 1}: Annual Income (k$) must be between 5 and 500.`);
    }
    if (!Number.isFinite(spending) || spending < 1 || spending > 100) {
      throw new Error(`Row ${r + 1}: Spending Score must be between 1 and 100.`);
    }

    let customerId = idIdx >= 0 ? Math.round(Number(cols[idIdx])) : r;
    if (!Number.isFinite(customerId) || customerId <= 0 || usedIds.has(customerId)) {
      customerId = r;
      while (usedIds.has(customerId)) customerId += 1;
    }
    usedIds.add(customerId);

    parsed.push({
      customer_id: customerId,
      gender,
      age: Math.round(age),
      annual_income_k: Number(income.toFixed(2)),
      spending_score: Number(spending.toFixed(2)),
    });
  }

  return parsed;
}

export function DataWorkspace({
  segments,
  activeCustomers,
  onPipelineUpdated,
  onNavigateBrief,
  onNavigateGeometry,
}: DataWorkspaceProps) {
  const [rows, setRows] = useState<ManualCustomerInput[]>(() =>
    customerRecordsToInputs(activeCustomers)
  );
  const [inputMode, setInputMode] = useState<'manual' | 'paste'>('manual');
  const [pasteText, setPasteText] = useState<string>(
    'CustomerID,Gender,Age,Annual Income (k$),Spending Score (1-100)\n' +
      '1,Female,24,24,78\n' +
      '2,Male,46,26,17\n' +
      '3,Female,32,88,86\n' +
      '4,Male,42,92,18\n' +
      '5,Female,40,55,50'
  );

  // Quick-add row inputs
  const [newGender, setNewGender] = useState<'Female' | 'Male'>('Female');
  const [newAge, setNewAge] = useState<number>(32);
  const [newIncome, setNewIncome] = useState<number>(75);
  const [newSpending, setNewSpending] = useState<number>(72);

  // Pipeline parameters
  const [targetK, setTargetK] = useState<number>(segments.target_k);
  const [includeGender, setIncludeGender] = useState<boolean>(segments.include_gender);
  const [entropyThreshold, setEntropyThreshold] = useState<number>(segments.entropy_threshold);

  // Pagination & status
  const [page, setPage] = useState<number>(0);
  const [busy, setBusy] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [successMsg, setSuccessMsg] = useState<string | null>(null);

  const pageSize = 20;
  const totalPages = Math.max(1, Math.ceil(rows.length / pageSize));
  const safePage = Math.min(page, totalPages - 1);
  const visibleRows = useMemo(
    () => rows.slice(safePage * pageSize, (safePage + 1) * pageSize),
    [rows, safePage]
  );

  const nextId = useMemo(() => {
    const maxId = rows.reduce((acc, r) => Math.max(acc, r.customer_id), 0);
    return maxId + 1;
  }, [rows]);

  const addManualRow = () => {
    setErrorMsg(null);
    setSuccessMsg(null);
    if (newAge < 16 || newAge > 100 || newIncome < 5 || newIncome > 500 || newSpending < 1 || newSpending > 100) {
      setErrorMsg('Enter Age (16-100), Income k$ (5-500), and Spending Score (1-100).');
      return;
    }
    const nextRows = [
      {
        customer_id: nextId,
        gender: newGender,
        age: Math.round(newAge),
        annual_income_k: Number(newIncome),
        spending_score: Number(newSpending),
      },
      ...rows,
    ];
    setRows(nextRows);
    setPage(0);
  };

  const updateRowField = (
    customerId: number,
    field: keyof ManualCustomerInput,
    value: string | number
  ) => {
    setRows((prev) =>
      prev.map((row) => (row.customer_id === customerId ? { ...row, [field]: value } : row))
    );
  };

  const deleteRow = (customerId: number) => {
    setRows((prev) => prev.filter((row) => row.customer_id !== customerId));
  };

  const handleLoadPaste = () => {
    setErrorMsg(null);
    setSuccessMsg(null);
    try {
      const parsed = parseDelimitedText(pasteText);
      setRows(parsed);
      setPage(0);
      setInputMode('manual');
      setSuccessMsg(`Loaded ${parsed.length} customer rows into the editable dataset table.`);
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : 'Unable to parse pasted text.');
    }
  };

  const handleFileUpload = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setErrorMsg(null);
    setSuccessMsg(null);
    try {
      const text = await file.text();
      const parsed = parseDelimitedText(text);
      setRows(parsed);
      setPage(0);
      setInputMode('manual');
      setSuccessMsg(
        `Imported ${parsed.length} customer rows from ${file.name}. Click "Run Segmentation Pipeline" to train & visualize.`
      );
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : 'Unable to read CSV file.');
    } finally {
      event.target.value = '';
    }
  };

  const handleDownloadCsv = () => {
    const header = 'CustomerID,Gender,Age,Annual Income (k$),Spending Score (1-100)';
    const body = rows.map(
      (r) => `${r.customer_id},${r.gender},${r.age},${r.annual_income_k},${r.spending_score}`
    );
    const blob = new Blob([[header, ...body].join('\n')], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'mall_customers_custom.csv';
    link.click();
    URL.revokeObjectURL(url);
  };

  const handleRunPipeline = async () => {
    setErrorMsg(null);
    setSuccessMsg(null);

    if (rows.length < 15) {
      setErrorMsg('At least 15 customer rows are required to fit PCA, K-Means, Ward, DBSCAN, and GMM.');
      return;
    }
    if (targetK >= rows.length) {
      setErrorMsg(`Target clusters (k=${targetK}) must be strictly smaller than ${rows.length} rows.`);
      return;
    }

    // Deduplicate IDs deterministically if manually edited to collide
    const seen = new Set<number>();
    const cleanRows = rows.map((r, idx) => {
      let cid = Math.max(1, Math.round(r.customer_id));
      while (seen.has(cid)) cid = idx + 1 + seen.size;
      seen.add(cid);
      return { ...r, customer_id: cid };
    });

    setBusy(true);
    try {
      const res = await fetch('/api/v1/train-custom', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          customers: cleanRows,
          target_k: targetK,
          include_gender: includeGender,
          entropy_threshold: entropyThreshold,
          enforce_stability_gate: false,
        }),
      });
      if (!res.ok) {
        let msg = `Pipeline training failed (HTTP ${res.status})`;
        try {
          const errJson = (await res.json()) as { detail?: string };
          if (errJson.detail) msg = errJson.detail;
        } catch {
          // ignore
        }
        throw new Error(msg);
      }
      const nextSegments = (await res.json()) as SegmentsPayload;
      await onPipelineUpdated(nextSegments);
      setSuccessMsg(
        `Segmented ${nextSegments.row_count} customers into ${nextSegments.target_k} personas · Silhouette ${nextSegments.silhouette_score.toFixed(3)} · Jaccard ${nextSegments.mean_jaccard_stability.toFixed(3)} · Run ${nextSegments.run_key.replace('mall-seg-', '')}`
      );
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : 'Segmentation pipeline failed.');
    } finally {
      setBusy(false);
    }
  };

  const handleResetDefault = async () => {
    setErrorMsg(null);
    setSuccessMsg(null);
    setBusy(true);
    try {
      const res = await fetch('/api/v1/reset-default', { method: 'POST' });
      if (!res.ok) {
        throw new Error(`Reset failed (HTTP ${res.status})`);
      }
      const nextSegments = (await res.json()) as SegmentsPayload;
      await onPipelineUpdated(nextSegments);
      setTargetK(nextSegments.target_k);
      setIncludeGender(nextSegments.include_gender);
      setEntropyThreshold(nextSegments.entropy_threshold);
      setSuccessMsg(
        `Restored default Mall Customers dataset (${nextSegments.row_count} rows, run ${nextSegments.run_key.replace('mall-seg-', '')}).`
      );
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : 'Unable to restore default dataset.');
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Pipeline Configuration & Action Bar */}
      <section className="bg-[#FFFFFF] border border-[#E4E2DC] rounded-xl p-5 space-y-5">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 border-b border-[#E4E2DC] pb-4">
          <div className="space-y-1">
            <h2 className="text-sm font-semibold text-[#141413]">
              Interactive Dataset Builder & End-to-End Pipeline Execution
            </h2>
            <p className="text-xs text-[#64635E]">
              Enter rows manually, paste CSV/TSV data, or upload a CSV file to run Pandera validation,
              PCA, K-Means++, Ward Hierarchical, DBSCAN, GMM entropy gating, and surrogate tree rules.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2 shrink-0">
            <label className="inline-flex items-center gap-1.5 px-3 py-2 text-xs font-medium bg-[#FAF9F6] hover:bg-[#EFECE5] border border-[#E4E2DC] rounded-lg cursor-pointer whitespace-nowrap shrink-0">
              <Upload className="w-3.5 h-3.5 text-[#1E40AF]" />
              <span>Import CSV File</span>
              <input
                type="file"
                accept=".csv,.tsv,.txt"
                onChange={(e) => void handleFileUpload(e)}
                className="hidden"
              />
            </label>

            <button
              type="button"
              onClick={handleDownloadCsv}
              className="inline-flex items-center gap-1.5 px-3 py-2 text-xs font-medium bg-[#FAF9F6] hover:bg-[#EFECE5] border border-[#E4E2DC] rounded-lg cursor-pointer whitespace-nowrap shrink-0"
            >
              <Download className="w-3.5 h-3.5" />
              <span>Export CSV</span>
            </button>

            <button
              type="button"
              disabled={busy}
              onClick={() => void handleResetDefault()}
              className="inline-flex items-center gap-1.5 px-3 py-2 text-xs font-medium bg-[#FAF9F6] hover:bg-[#EFECE5] border border-[#E4E2DC] rounded-lg cursor-pointer whitespace-nowrap shrink-0 disabled:opacity-50"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              <span>Reset Default</span>
            </button>

            <button
              type="button"
              disabled={busy}
              onClick={() => void handleRunPipeline()}
              className="inline-flex items-center gap-1.5 px-4 py-2 text-xs font-semibold bg-[#1E40AF] text-white hover:bg-[#1E3A8A] rounded-lg cursor-pointer whitespace-nowrap shrink-0 disabled:opacity-60"
            >
              <Play className="w-3.5 h-3.5" />
              <span>{busy ? 'Running ML Pipeline…' : `Run Pipeline (${rows.length} rows)`}</span>
            </button>
          </div>
        </div>

        {/* Hyperparameters & Privacy Opt-In Controls */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="p-3.5 bg-[#FAF9F6] border border-[#E4E2DC] rounded-lg space-y-2">
            <div className="flex justify-between text-xs">
              <span className="text-[#5C5B56] font-medium">Target Segments (k)</span>
              <span className="font-mono-tabular font-semibold text-[#141413]">{targetK}</span>
            </div>
            <input
              type="range"
              min={2}
              max={10}
              value={targetK}
              onChange={(e) => setTargetK(Number(e.target.value))}
              className="w-full accent-[#1E40AF]"
            />
          </div>

          <div className="p-3.5 bg-[#FAF9F6] border border-[#E4E2DC] rounded-lg space-y-2">
            <div className="flex justify-between text-xs">
              <span className="text-[#5C5B56] font-medium">Boundary Entropy Gate</span>
              <span className="font-mono-tabular font-semibold text-[#141413]">
                {entropyThreshold.toFixed(2)}
              </span>
            </div>
            <input
              type="range"
              min={0.2}
              max={0.8}
              step={0.05}
              value={entropyThreshold}
              onChange={(e) => setEntropyThreshold(Number(e.target.value))}
              className="w-full accent-[#1E40AF]"
            />
          </div>

          <div className="p-3.5 bg-[#FAF9F6] border border-[#E4E2DC] rounded-lg flex flex-col justify-between gap-2">
            <span className="text-xs text-[#5C5B56] font-medium">
              Gender Distance Inclusion (Privacy Opt-In)
            </span>
            <div className="grid grid-cols-2 gap-1.5">
              <button
                type="button"
                onClick={() => setIncludeGender(false)}
                className={`py-1.5 px-2 text-xs rounded-md border cursor-pointer whitespace-nowrap ${
                  !includeGender
                    ? 'bg-[#141413] text-white border-[#141413] font-medium'
                    : 'bg-[#FFFFFF] text-[#5C5B56] border-[#E4E2DC]'
                }`}
              >
                Profiling Only
              </button>
              <button
                type="button"
                onClick={() => setIncludeGender(true)}
                className={`py-1.5 px-2 text-xs rounded-md border cursor-pointer whitespace-nowrap ${
                  includeGender
                    ? 'bg-[#141413] text-white border-[#141413] font-medium'
                    : 'bg-[#FFFFFF] text-[#5C5B56] border-[#E4E2DC]'
                }`}
              >
                Include Gender
              </button>
            </div>
          </div>
        </div>

        {errorMsg && (
          <div className="flex items-start gap-2 p-3.5 bg-[#BE123C]/5 border border-[#BE123C]/30 rounded-lg text-xs text-[#BE123C]">
            <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
            <span>{errorMsg}</span>
          </div>
        )}

        {successMsg && (
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 p-3.5 bg-[#047857]/5 border border-[#047857]/30 rounded-lg text-xs text-[#047857]">
            <div className="flex items-center gap-2 font-mono-tabular">
              <CheckCircle2 className="w-4 h-4 shrink-0" />
              <span>{successMsg}</span>
            </div>
            <div className="flex items-center gap-2 shrink-0">
              <button
                type="button"
                onClick={onNavigateBrief}
                className="px-2.5 py-1 bg-[#FFFFFF] border border-[#047857]/30 rounded text-xs font-medium text-[#047857] hover:bg-[#FAF9F6] cursor-pointer whitespace-nowrap"
              >
                View Brief
              </button>
              <button
                type="button"
                onClick={onNavigateGeometry}
                className="px-2.5 py-1 bg-[#047857] text-white rounded text-xs font-medium hover:bg-[#065F46] cursor-pointer whitespace-nowrap"
              >
                Open Cluster Geometry
              </button>
            </div>
          </div>
        )}
      </section>

      {/* Mode Switcher + Dataset Table / Paste Editor */}
      <section className="bg-[#FFFFFF] border border-[#E4E2DC] rounded-xl overflow-hidden">
        <div className="px-5 py-3.5 border-b border-[#E4E2DC] flex flex-wrap items-center justify-between gap-3 bg-[#FAF9F6]">
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => setInputMode('manual')}
              className={`px-3 py-1.5 text-xs font-medium rounded-md cursor-pointer whitespace-nowrap ${
                inputMode === 'manual'
                  ? 'bg-[#141413] text-white'
                  : 'bg-[#FFFFFF] border border-[#E4E2DC] text-[#5C5B56]'
              }`}
            >
              Manual Table Editor ({rows.length} rows)
            </button>
            <button
              type="button"
              onClick={() => setInputMode('paste')}
              className={`px-3 py-1.5 text-xs font-medium rounded-md cursor-pointer whitespace-nowrap ${
                inputMode === 'paste'
                  ? 'bg-[#141413] text-white'
                  : 'bg-[#FFFFFF] border border-[#E4E2DC] text-[#5C5B56]'
              }`}
            >
              Paste CSV / TSV Text
            </button>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={() => {
                setRows(customerRecordsToInputs(activeCustomers));
                setPage(0);
                setErrorMsg(null);
              }}
              className="px-2.5 py-1.5 text-xs bg-[#FFFFFF] border border-[#E4E2DC] rounded-md text-[#5C5B56] hover:text-[#141413] cursor-pointer whitespace-nowrap"
            >
              Load Active Cohort
            </button>
            <button
              type="button"
              onClick={() => {
                setRows(MANUAL_TEMPLATE_ROWS);
                setPage(0);
                setErrorMsg(null);
                setSuccessMsg('Loaded 30-row multi-segment manual starter template.');
              }}
              className="px-2.5 py-1.5 text-xs bg-[#FFFFFF] border border-[#E4E2DC] rounded-md text-[#5C5B56] hover:text-[#141413] cursor-pointer whitespace-nowrap"
            >
              Load 30-Row Starter
            </button>
          </div>
        </div>

        {inputMode === 'paste' ? (
          <div className="p-5 space-y-3">
            <div className="flex items-center gap-2 text-xs text-[#5C5B56]">
              <FileSpreadsheet className="w-4 h-4 text-[#1E40AF]" />
              <span>
                Paste CSV or tab-separated rows with header columns:{' '}
                <span className="font-mono-tabular text-[#141413]">
                  CustomerID, Gender, Age, Annual Income (k$), Spending Score (1-100)
                </span>
              </span>
            </div>
            <textarea
              value={pasteText}
              onChange={(e) => setPasteText(e.target.value)}
              rows={10}
              className="w-full p-3.5 font-mono-tabular text-xs bg-[#FAF9F6] border border-[#E4E2DC] rounded-lg text-[#141413] focus:outline-none focus:border-[#1E40AF]"
            />
            <div className="flex justify-end">
              <button
                type="button"
                onClick={handleLoadPaste}
                className="px-4 py-2 text-xs font-semibold bg-[#141413] text-white rounded-lg hover:bg-[#2B2B29] cursor-pointer"
              >
                Parse & Load into Editor
              </button>
            </div>
          </div>
        ) : (
          <div>
            {/* Quick Add Row Bar */}
            <div className="p-4 border-b border-[#E4E2DC] bg-[#FFFFFF] grid grid-cols-2 sm:grid-cols-6 gap-3 items-end">
              <div>
                <label className="block text-[11px] text-[#64635E] mb-1">Next ID</label>
                <div className="px-2.5 py-1.5 text-xs font-mono-tabular bg-[#FAF9F6] border border-[#E4E2DC] rounded-md text-[#64635E]">
                  {nextId}
                </div>
              </div>
              <div>
                <label className="block text-[11px] text-[#64635E] mb-1">Gender</label>
                <select
                  value={newGender}
                  onChange={(e) => setNewGender(e.target.value as 'Male' | 'Female')}
                  className="w-full px-2.5 py-1.5 text-xs bg-[#FAF9F6] border border-[#E4E2DC] rounded-md text-[#141413]"
                >
                  <option value="Female">Female</option>
                  <option value="Male">Male</option>
                </select>
              </div>
              <div>
                <label className="block text-[11px] text-[#64635E] mb-1">Age (16–100)</label>
                <input
                  type="number"
                  min={16}
                  max={100}
                  value={newAge}
                  onChange={(e) => setNewAge(Number(e.target.value))}
                  className="w-full px-2.5 py-1.5 text-xs font-mono-tabular bg-[#FAF9F6] border border-[#E4E2DC] rounded-md text-[#141413]"
                />
              </div>
              <div>
                <label className="block text-[11px] text-[#64635E] mb-1">Income (k$)</label>
                <input
                  type="number"
                  min={5}
                  max={500}
                  value={newIncome}
                  onChange={(e) => setNewIncome(Number(e.target.value))}
                  className="w-full px-2.5 py-1.5 text-xs font-mono-tabular bg-[#FAF9F6] border border-[#E4E2DC] rounded-md text-[#141413]"
                />
              </div>
              <div>
                <label className="block text-[11px] text-[#64635E] mb-1">Spending (1–100)</label>
                <input
                  type="number"
                  min={1}
                  max={100}
                  value={newSpending}
                  onChange={(e) => setNewSpending(Number(e.target.value))}
                  className="w-full px-2.5 py-1.5 text-xs font-mono-tabular bg-[#FAF9F6] border border-[#E4E2DC] rounded-md text-[#141413]"
                />
              </div>
              <button
                type="button"
                onClick={addManualRow}
                className="inline-flex items-center justify-center gap-1.5 px-3 py-1.5 text-xs font-semibold bg-[#141413] text-white rounded-md hover:bg-[#2B2B29] cursor-pointer whitespace-nowrap"
              >
                <Plus className="w-3.5 h-3.5" />
                <span>Add Customer</span>
              </button>
            </div>

            {/* Editable Spreadsheet Table */}
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-[#E4E2DC] bg-[#FAF9F6] text-[#5C5B56]">
                    <th className="py-2.5 px-4 font-medium">CustomerID</th>
                    <th className="py-2.5 px-3 font-medium">Gender</th>
                    <th className="py-2.5 px-3 font-medium">Age</th>
                    <th className="py-2.5 px-3 font-medium">Annual Income (k$)</th>
                    <th className="py-2.5 px-3 font-medium">Spending Score (1–100)</th>
                    <th className="py-2.5 px-4 text-right font-medium">Remove</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#E4E2DC]">
                  {visibleRows.map((row) => (
                    <tr key={row.customer_id} className="hover:bg-[#FAF9F6]">
                      <td className="py-2 px-4 font-mono-tabular text-[#64635E]">
                        {row.customer_id}
                      </td>
                      <td className="py-2 px-3">
                        <select
                          value={row.gender}
                          onChange={(e) =>
                            updateRowField(row.customer_id, 'gender', e.target.value)
                          }
                          className="px-2 py-1 text-xs bg-[#FFFFFF] border border-[#E4E2DC] rounded text-[#141413]"
                        >
                          <option value="Female">Female</option>
                          <option value="Male">Male</option>
                        </select>
                      </td>
                      <td className="py-2 px-3">
                        <input
                          type="number"
                          min={16}
                          max={100}
                          value={row.age}
                          onChange={(e) =>
                            updateRowField(row.customer_id, 'age', Number(e.target.value))
                          }
                          className="w-20 px-2 py-1 text-xs font-mono-tabular bg-[#FFFFFF] border border-[#E4E2DC] rounded text-[#141413]"
                        />
                      </td>
                      <td className="py-2 px-3">
                        <input
                          type="number"
                          min={5}
                          max={500}
                          value={row.annual_income_k}
                          onChange={(e) =>
                            updateRowField(
                              row.customer_id,
                              'annual_income_k',
                              Number(e.target.value)
                            )
                          }
                          className="w-24 px-2 py-1 text-xs font-mono-tabular bg-[#FFFFFF] border border-[#E4E2DC] rounded text-[#141413]"
                        />
                      </td>
                      <td className="py-2 px-3">
                        <input
                          type="number"
                          min={1}
                          max={100}
                          value={row.spending_score}
                          onChange={(e) =>
                            updateRowField(
                              row.customer_id,
                              'spending_score',
                              Number(e.target.value)
                            )
                          }
                          className="w-24 px-2 py-1 text-xs font-mono-tabular bg-[#FFFFFF] border border-[#E4E2DC] rounded text-[#141413]"
                        />
                      </td>
                      <td className="py-2 px-4 text-right">
                        <button
                          type="button"
                          onClick={() => deleteRow(row.customer_id)}
                          aria-label={`Delete customer ${row.customer_id}`}
                          className="p-1 text-[#8A8880] hover:text-[#BE123C] cursor-pointer"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Pagination Footer */}
            <div className="px-5 py-3 border-t border-[#E4E2DC] bg-[#FAF9F6] flex items-center justify-between text-xs text-[#5C5B56]">
              <span className="font-mono-tabular">
                Showing {rows.length === 0 ? 0 : safePage * pageSize + 1}–
                {Math.min((safePage + 1) * pageSize, rows.length)} of {rows.length} rows
                {rows.length < 15 ? ' (add at least 15 rows to train)' : ''}
              </span>
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  disabled={safePage === 0}
                  onClick={() => setPage((p) => Math.max(0, p - 1))}
                  className="px-2.5 py-1 bg-[#FFFFFF] border border-[#E4E2DC] rounded disabled:opacity-40 cursor-pointer"
                >
                  Prev
                </button>
                <span className="font-mono-tabular">
                  {safePage + 1} / {totalPages}
                </span>
                <button
                  type="button"
                  disabled={safePage >= totalPages - 1}
                  onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
                  className="px-2.5 py-1 bg-[#FFFFFF] border border-[#E4E2DC] rounded disabled:opacity-40 cursor-pointer"
                >
                  Next
                </button>
              </div>
            </div>
          </div>
        )}
      </section>
    </div>
  );
}
