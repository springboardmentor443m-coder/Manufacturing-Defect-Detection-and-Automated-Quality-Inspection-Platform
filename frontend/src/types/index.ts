export type UserRole = 'quality_engineer' | 'supervisor' | 'admin';

export interface User {
  id: number;
  email: string;
  full_name: string | null;
  role: UserRole;
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

export interface AuthResponse {
  access_token: string;
  token_type: string;
  role: UserRole;
  user_id: number;
  email: string;
  full_name: string | null;
}

export interface Product {
  id: number;
  name: string;
  sku: string;
  description: string | null;
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

export interface Defect {
  id: number;
  inspection_id: number;
  defect_type: string;
  confidence: number;
  bounding_box?: Record<string, any> | null;
  segmentation_mask_path?: string | null;
  size_score: number;
  location_score: number;
  defect_type_score: number;
  confidence_score: number;
  severity_score: number;
  created_at: string;
}

export interface Report {
  id: number;
  inspection_id: number;
  summary: string | null;
  notes: string | null;
  generated_by: number | null;
  report_data?: Record<string, any> | null;
  created_at: string;
}

export interface QualityMetricDetail {
  value: number;
  status: 'pass' | 'warn' | 'fail';
  label: string;
  unit: string;
}

export interface QualityReport {
  inspection_id?: number;
  verdict: 'pass' | 'warn' | 'fail';
  overall_score: number;
  reasons: string[];
  resolution: {
    width: number;
    height: number;
    megapixels?: number;
    aspect_ratio?: number;
  };
  blur_score: number;
  brightness: number;
  contrast: number;
  noise_estimate: number;
  metrics_breakdown: Record<string, QualityMetricDetail>;
  features?: {
    edges?: {
      edge_pixel_count: number;
      edge_density_pct: number;
      mean_gradient_magnitude: number;
    };
    texture?: {
      mean_intensity: number;
      std_intensity: number;
      variance: number;
      entropy: number;
      uniformity_energy: number;
      skewness: number;
    };
    shape?: {
      contour_count: number;
      dominant_area_px: number;
      dominant_perimeter_px: number;
      circularity: number;
      solidity: number;
      aspect_ratio: number;
      bounding_box?: { x: number; y: number; width: number; height: number };
    };
    edge_preview_data_url?: string | null;
  };
  original_image_url?: string;
  processed_image_url?: string | null;
}

export interface PreprocessingConfig {
  do_validate: boolean;
  do_resize: boolean;
  target_width: number;
  target_height: number;
  keep_aspect_ratio: boolean;
  do_denoise: boolean;
  denoise_method: 'gaussian' | 'median' | 'bilateral';
  denoise_kernel_size: number;
  do_clahe: boolean;
  clahe_clip_limit: number;
  clahe_grid_size: number;
  do_edge_detection: boolean;
  edge_low_threshold: number;
  edge_high_threshold: number;
  do_roi_extraction: boolean;
  roi_mode: 'center' | 'bbox' | 'auto_foreground';
  roi_margin_percent: number;
}

export interface PreprocessResponse {
  inspection_id: number;
  message: string;
  processed_image_url: string;
  processed_image_path: string;
  quality_verdict: 'pass' | 'warn' | 'fail';
  quality_score: number;
  pipeline_metadata: Record<string, any>;
  quality_report: QualityReport;
}

export interface Inspection {
  id: number;
  product_id: number;
  image_path: string;
  original_filename?: string | null;
  file_size_bytes?: number | null;
  image_width?: number | null;
  image_height?: number | null;
  mime_type?: string | null;
  image_url?: string | null;
  processed_image_path?: string | null;
  processed_image_url?: string | null;
  quality_score?: number | null;
  quality_verdict?: 'pass' | 'warn' | 'fail' | null;
  quality_report?: QualityReport | null;
  preprocessing_config?: Record<string, any> | null;
  extra_metadata?: Record<string, any> | null;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  decision: 'pass' | 'fail' | 'rework' | 'manual_review' | null;
  overall_severity: number;
  created_by: number;
  created_at: string;
  updated_at: string;
  defects?: Defect[];
  reports?: Report[];
}

export interface InspectionListResponse {
  items: Inspection[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

export interface UploadErrorItem {
  filename: string;
  error: string;
}

export interface BatchUploadResponse {
  total: number;
  successful: number;
  failed: number;
  inspections: Inspection[];
  errors: UploadErrorItem[];
}
