"""
Earthquake Observation Data Mapping Script
Maps Napa (2014) and Ridgecrest (2019) datasets to Current schema format
"""

import pandas as pd
import numpy as np
import warnings
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EarthquakeDataMapper:
    """Class to handle mapping of earthquake observation data to current schema"""
    
    def __init__(self):
        self.napa_mapping = {
            'stnid': 'Station_ID', 
            'intid': None, 
            'observer': 'Creator',  
            'obs_date': 'Date_of_Movement',  
            'origin': 'Feature_Origin',  
            'latitude': '_latitude', 
            'longitude': '_longitude',  
            'orig_lat': '_orig_lat', 
            'orig_lon': '_orig_lon', 
            'description': 'Notes',
            'citation': '_citation', 
            'photo': '_photo', 
            'fault_azimuth': 'Local_Fault_Azimuth_Degrees',
            'ss_displacement': 'Horizontal_Separation_cm',
            'ss_sense': 'Slip_Sense',
            'ext_offset': 'Heave_cm', 
            'comp_offset': '_comp_offset', 
            'vert_offset': 'Vertical_Separation_cm',
            'upthrown_side': 'Scarp_Facing_Direction',
            'observed_feature': '_observed_feature', 
            'trace': '_trace', 
        }
        
        self.ridgecrest_mapping = {
            'intid' : None,  # Let OBJECTID auto-generate
            'origid' : 'Station_ID',
            'observer' : 'Creator',
            'obs_affiliation' : '_obs_affiliation',  
            'team_id' : '_team_id',  
            'team' : '_team',  
            'obs_position' : '_obs_position',  
            'obs_date' : 'CreationDate',
            'origin' : 'Feature_Origin',
            'source' : '_source',  
            'citation' : '_citation',  
            'description' : 'Notes',
            'fault_az_min' : '_fault_az_min', 
            'fault_az_pref' : 'Local_Fault_Azimuth_Degrees',
            'fault_az_max' : '_fault_az_max',  
            'fault_dip_min' : '_fault_dip_min',
            'fault_dip_pref' : 'Local_Fault_Dip',
            'fault_dip_max' : '_fault_dip_max',  
            'local_frac_az_min' : '_local_frac_az_min',  
            'local_frac_az_pref' : '_local_frac_az_pref',
            'local_frac_az_max' : '_local_frac_az_max', 
            'rup_width_min' : 'Rupture_Width_Min_m',
            'rupture_width_pref' : 'Rupture_Width_m',
            'rup_width_max' : 'Rupture_Width_Max_m',
            'fault_expression' : 'Rupture_Expression',
            'scarp_facing_direction' : 'Scarp_Facing_Direction',
            'striations_observed' : '_striations_observed',
            'gouge_observed' : '_gouge_observed',
            'sense' : 'Slip_Sense',
            'observed_feature' : '_observed_feature',
            'feature_type' : '_feature_type',
            'vector_length_min' : 'Net_Slip_Min_cm',
            'vector_length_pref' : 'Net_Slip_Preferred_cm',
            'vector_length_max' : 'Net_Slip_Max_cm',
            'vect_plunge_min' : '_vect_plunge_min',
            'vect_plunge_pref' : 'Plunge',
            'vect_plunge_max' : '_vect_plunge_max',
            'vect_az_min' : '_vect_az_min',
            'vect_az_pref' : 'VM_Slip_Azimuth',
            'vect_az_max' : '_vect_az_max',
            'aperture_min' : '_aperture_min',
            'aperture_pref' : '_aperture_pref',
            'aperture_max' : '_aperture_max',
            'horiz_offset_min' : 'Horizontal_Separation_Min_cm',
            'horiz_offset_pref' : 'Horizontal_Separation_cm',
            'horiz_offset_max' : 'Horizontal_Separation_Max_cm',
            'horiz_slip_type' : 'Fault_Slip_Measurement_Type',
            'horiz_az_min' : '_horiz_az_min', 
            'horiz_az_pref' : 'Slip_Azimuth',
            'horiz_az_max' : '_horiz_az_max',
            'vert_offset_min' : 'Vertical_Separation_Min_cm',
            'vert_offset_pref' : 'Vertical_Separation_cm',
            'vert_offset_max' : 'Vertical_Separation_Max_cm',
            'vert_slip_type' : '_vert_slip_type',
            'heave_type' : '_heave_type',
            'heave_min' : 'Heave_min_cm',
            'heave_pref' : 'Heave_cm',
            'heave_max' : 'Heave_max_cm',
            'latitude' : '_latitude', 
            'longitude' : '_longitude',
            'orig_lat' : '_orig_lat',
            'orig_lon' : '_orig_lon',
            'note' : 'Vector_Offset_Feature_Notes',
        }
        
        # Define current schema structure (base fields)
        self.current_schema_fields = [
            'OBJECTID', 'Station_ID', 'Feature_Origin', 'Notes', 'Confidence_Feature_ID',
            'Mode_Observation', 'Slip_Sense', 'Scarp_Facing_Direction', 'Local_Fault_Dip',
            'Local_Fault_Azimuth_Degrees', 'Slip_Azimuth', 'Fault_Slip_Measurement_Type',
            'Heave_cm', 'Heave_min_cm', 'Heave_max_cm', 'Rupture_Expression',
            'Rupture_Width_m', 'Rupture_Width_Min_m', 'Rupture_Width_Max_m',
            'VM_Slip_Azimuth', 'Plunge', 'Net_Slip_Preferred_cm', 'Net_Slip_Min_cm',
            'Net_Slip_Max_cm', 'Vector_Measurement_Confidence', 'Vector_Offset_Feature_Notes',
            'Horizontal_Separation_cm', 'Horizontal_Separation_Min_cm', 'Horizontal_Separation_Max_cm',
            'Vertical_Separation_cm', 'Vertical_Separation_Min_cm', 'Vertical_Separation_Max_cm',
            'Slip_Measurement_Confidence', 'Slip_Offset_Feature_Notes', 'Diameter_m',
            'Height_of_Material_m', 'Estimated_Max_VertMov_m', 'LQ_Area_Affected_sqm',
            'Date_of_Movement', 'Displacement_Amount', 'Est_Direction_SM', 'Landslide_Feature',
            'Slide_Type', 'Material_Type', 'Depth', 'SM_Area_Affected_sqm',
            'Est_Max_Drop_Elev_ft', 'Length_Exposed_Downslope', 'Cause_of_Damage',
            'Facility_Affected', 'Utility_Affected', 'Damage_Severity', 'GlobalID',
            'CreationDate', 'Creator', 'EditDate', 'Editor'
        ]

    def get_extended_schema_fields(self, mapping):
        """Get extended schema including new columns for unmapped fields"""
        # Start with base current schema
        extended_fields = self.current_schema_fields.copy()
        
        # Add all new columns (those starting with _) from mapping
        new_columns = [field for field in mapping.values() if field and field.startswith('_')]
        new_columns = sorted(list(set(new_columns)))  # Remove duplicates and sort
        
        # Add new columns at the end
        extended_fields.extend(new_columns)
        
        return extended_fields

    def load_napa_data(self, file_path):
        """Load Napa dataset"""
        try:
            if file_path.endswith('.csv'):
                # Try different encodings
                for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding)
                        
                        # If only 1 column, try tab-separated
                        if len(df.columns) == 1:
                            df = pd.read_csv(file_path, sep='\t', encoding=encoding)
                        
                        logger.info(f"Successfully loaded Napa data with {encoding} encoding")
                        break
                    except UnicodeDecodeError:
                        continue
                    except Exception as e:
                        logger.warning(f"Failed with {encoding}: {e}")
                        continue
            else:
                df = pd.read_excel(file_path)
                
            logger.info(f"Loaded Napa data: {len(df)} records, {len(df.columns)} columns")
            return df
        except Exception as e:
            logger.error(f"Error loading Napa data: {e}")
            return None

    def load_ridgecrest_data(self, file_path):
        """Load Ridgecrest dataset"""
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
            logger.info(f"Loaded Ridgecrest data: {len(df)} records, {len(df.columns)} columns")
            return df
        except Exception as e:
            logger.error(f"Error loading Ridgecrest data: {e}")
            return None

    def map_napa_to_current(self, napa_df):
        """Map Napa dataset to current schema"""
        logger.info("Mapping Napa data to current schema...")
        
        # Get extended schema including new columns
        extended_fields = self.get_extended_schema_fields(self.napa_mapping)
        current_df = pd.DataFrame(columns=extended_fields)
        
        for idx, row in napa_df.iterrows():
            mapped_row = {}
            
            # Generate OBJECTID (auto-incrementing)
            mapped_row['OBJECTID'] = idx + 1
            
            # Apply ALL mappings (both direct and new columns)
            for napa_field, current_field in self.napa_mapping.items():
                if current_field and napa_field in row:
                    mapped_row[current_field] = row[napa_field]
            
            # Set default values for system fields
            mapped_row['CreationDate'] = datetime.now()
            mapped_row['EditDate'] = datetime.now()
            mapped_row['Editor'] = 'Napa_Migration'
            
            # Add dataset source identifier to Notes
            existing_notes = mapped_row.get('Notes', '')
            if existing_notes:
                mapped_row['Notes'] = f"Source: Napa 2014; {existing_notes}"
            else:
                mapped_row['Notes'] = "Source: Napa 2014"
            
            # Append to dataframe
            current_df = pd.concat([current_df, pd.DataFrame([mapped_row])], ignore_index=True)
        
        logger.info(f"Mapped {len(current_df)} Napa records to extended current schema")
        return current_df

    def map_ridgecrest_to_current(self, ridgecrest_df):
        """Map Ridgecrest dataset to current schema"""
        logger.info("Mapping Ridgecrest data to current schema...")
        
        # Get extended schema including new columns
        extended_fields = self.get_extended_schema_fields(self.ridgecrest_mapping)
        current_df = pd.DataFrame(columns=extended_fields)
        
        for idx, row in ridgecrest_df.iterrows():
            mapped_row = {}
            
            # Generate OBJECTID (auto-incrementing, offset to avoid conflicts with Napa)
            mapped_row['OBJECTID'] = idx + 1
            
            # Apply ALL mappings (both direct and new columns)
            for ridgecrest_field, current_field in self.ridgecrest_mapping.items():
                if current_field and ridgecrest_field in row:
                    mapped_row[current_field] = row[ridgecrest_field]
            
            # Set default values for system fields
            mapped_row['CreationDate'] = datetime.now()
            mapped_row['EditDate'] = datetime.now() 
            mapped_row['Editor'] = 'Ridgecrest_Migration'
            
            # Add dataset source identifier to Notes
            existing_notes = mapped_row.get('Notes', '')
            if existing_notes:
                mapped_row['Notes'] = f"Source: Ridgecrest 2019; {existing_notes}"
            else:
                mapped_row['Notes'] = "Source: Ridgecrest 2019"
            
            # Append to dataframe
            current_df = pd.concat([current_df, pd.DataFrame([mapped_row])], ignore_index=True)
        
        logger.info(f"Mapped {len(current_df)} Ridgecrest records to extended current schema")
        return current_df

    def consolidate_datasets(self, napa_current_df, ridgecrest_current_df):
        """Consolidate mapped datasets into single current schema dataset"""
        logger.info("Consolidating datasets...")
        
        # Get all unique columns from both datasets
        all_columns = list(set(napa_current_df.columns.tolist() + ridgecrest_current_df.columns.tolist()))
        all_columns = sorted(all_columns)  # Sort for consistency
        
        # Ensure both dataframes have the same columns (fill missing with NaN)
        for col in all_columns:
            if col not in napa_current_df.columns:
                napa_current_df[col] = np.nan
            if col not in ridgecrest_current_df.columns:
                ridgecrest_current_df[col] = np.nan
        
        # Reorder columns
        napa_current_df = napa_current_df[all_columns]
        ridgecrest_current_df = ridgecrest_current_df[all_columns]
        
        # Combine datasets
        consolidated_df = pd.concat([napa_current_df, ridgecrest_current_df], ignore_index=True)
        
        # Reset OBJECTID to be sequential
        consolidated_df['OBJECTID'] = range(1, len(consolidated_df) + 1)
        
        # Fill NaN values appropriately
        for col in consolidated_df.columns:
            if col in ['Notes', 'Vector_Offset_Feature_Notes', 'Slip_Offset_Feature_Notes'] or col.startswith('_'):
                consolidated_df[col] = consolidated_df[col].fillna('')
            elif consolidated_df[col].dtype == 'object':
                consolidated_df[col] = consolidated_df[col].fillna('')
            else:
                consolidated_df[col] = consolidated_df[col].fillna(np.nan)
        
        logger.info(f"Consolidated dataset: {len(consolidated_df)} total records with {len(all_columns)} columns")
        return consolidated_df

    def finalize_single_dataset(self, df):
        """Finalize single dataset by filling NaN values appropriately"""
        for col in df.columns:
            if col in ['Notes', 'Vector_Offset_Feature_Notes', 'Slip_Offset_Feature_Notes'] or col.startswith('_'):
                df[col] = df[col].fillna('')
            elif df[col].dtype == 'object':
                df[col] = df[col].fillna('')
            else:
                df[col] = df[col].fillna(np.nan)
        return df

    def generate_migration_report(self, napa_df, ridgecrest_df, napa_current, ridgecrest_current, consolidated_df):
        """Generate report on data migration"""
        report = []
        report.append("EARTHQUAKE DATA MIGRATION REPORT")
        report.append("=" * 50)
        report.append(f"Migration Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        report.append("INPUT DATASETS:")
        if napa_df is not None:
            report.append(f"  Napa (2014): {len(napa_df)} records, {len(napa_df.columns)} fields")
        if ridgecrest_df is not None:
            report.append(f"  Ridgecrest (2019): {len(ridgecrest_df)} records, {len(ridgecrest_df.columns)} fields")
        report.append("")
        
        report.append("OUTPUT DATASETS:")
        if napa_current is not None:
            report.append(f"  Napa mapped: {len(napa_current)} records, {len(napa_current.columns)} columns")
        if ridgecrest_current is not None:
            report.append(f"  Ridgecrest mapped: {len(ridgecrest_current)} records, {len(ridgecrest_current.columns)} columns")
        if consolidated_df is not None:
            report.append(f"  Consolidated: {len(consolidated_df)} records, {len(consolidated_df.columns)} columns")
        report.append("")
        
        report.append("STRUCTURAL DATA PRESERVATION:")
        report.append("  - Location data preserved in dedicated columns (_latitude, _longitude)")
        report.append("  - Observer metadata preserved in structured format (_obs_affiliation, _team_id, etc.)")
        report.append("  - Measurement ranges preserved (_fault_az_min, _fault_az_max, etc.)")
        report.append("  - Observational details preserved (_striations_observed, _gouge_observed, etc.)")
        report.append("  - All new columns marked with underscore prefix for identification")
        report.append("")
        
        report.append("FIELD MAPPING SUMMARY:")
        if napa_df is not None:
            napa_direct = sum(1 for v in self.napa_mapping.values() if v is not None and not v.startswith('_'))
            napa_new = sum(1 for v in self.napa_mapping.values() if v is not None and v.startswith('_'))
            report.append(f"  Napa - Direct mappings: {napa_direct}, New columns: {napa_new}")
        
        if ridgecrest_df is not None:
            ridge_direct = sum(1 for v in self.ridgecrest_mapping.values() if v is not None and not v.startswith('_'))
            ridge_new = sum(1 for v in self.ridgecrest_mapping.values() if v is not None and v.startswith('_'))
            report.append(f"  Ridgecrest - Direct mappings: {ridge_direct}, New columns: {ridge_new}")
        report.append("")
        
        report.append("CRITICAL IMPROVEMENTS:")
        report.append("  - FIXED: Location data now in dedicated columns (was in Notes)")
        report.append("  - ENHANCED: All data preserved as structured data (not unstructured text)")
        report.append("  - ORGANIZED: New columns clearly marked with underscore prefix")
        report.append("  - MAINTAINED: Original Current schema structure intact")
        report.append("")
        
        return "\n".join(report)

def main():
    """Main execution function"""
    # Initialize mapper
    mapper = EarthquakeDataMapper()
    
    # File paths (update these to your actual file locations)
    napa_file = "C:/Users/rajuv/OneDrive/Desktop/Work/SCEC SOURCES Internship/SCEC/Mapping to Current/napa_observations.csv" 
    ridgecrest_file = "C:/Users/rajuv/OneDrive/Desktop/Work/SCEC SOURCES Internship/SCEC/Mapping to Current/ridgecrest_observations.csv"  
    
    try:
        # Load datasets
        logger.info("Starting earthquake data migration...")
        napa_df = mapper.load_napa_data(napa_file)
        ridgecrest_df = mapper.load_ridgecrest_data(ridgecrest_file)
        
        # Check if loading failed
        if napa_df is None and ridgecrest_df is None:
            logger.error("Failed to load both datasets")
            return
        elif napa_df is None:
            logger.warning("Failed to load Napa data - proceeding with Ridgecrest only")
        elif ridgecrest_df is None:
            logger.warning("Failed to load Ridgecrest data - proceeding with Napa only")
        
        # Map to current schema
        napa_current = None
        ridgecrest_current = None
        
        if napa_df is not None:
            napa_current = mapper.map_napa_to_current(napa_df)
            napa_current = mapper.finalize_single_dataset(napa_current)
        
        if ridgecrest_df is not None:
            ridgecrest_current = mapper.map_ridgecrest_to_current(ridgecrest_df)
            ridgecrest_current = mapper.finalize_single_dataset(ridgecrest_current)
        
        # Save individual datasets
        date_str = datetime.now().strftime('%Y%m%d')
        
        if napa_current is not None:
            napa_output_file = f"napa_current_schema_{date_str}.csv"
            napa_current.to_csv(napa_output_file, index=False)
            logger.info(f"Napa dataset saved to: {napa_output_file}")
        
        if ridgecrest_current is not None:
            ridgecrest_output_file = f"ridgecrest_current_schema_{date_str}.csv"
            ridgecrest_current.to_csv(ridgecrest_output_file, index=False)
            logger.info(f"Ridgecrest dataset saved to: {ridgecrest_output_file}")
        
        # Create consolidated dataset (optional)
        consolidated = None
        if napa_current is not None and ridgecrest_current is not None:
            consolidated = mapper.consolidate_datasets(napa_current, ridgecrest_current)
            consolidated_output_file = f"consolidated_earthquake_observations_{date_str}.csv"
            consolidated.to_csv(consolidated_output_file, index=False)
            logger.info(f"Consolidated dataset saved to: {consolidated_output_file}")
        
        # Generate and save report
        report = mapper.generate_migration_report(napa_df, ridgecrest_df, napa_current, ridgecrest_current, consolidated)
        report_file = f"migration_report_{date_str}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"Migration report saved to: {report_file}")
        
        print("\nMigration completed successfully!")
        output_files = []
        if napa_current is not None:
            output_files.append(f"napa_current_schema_{date_str}.csv")
        if ridgecrest_current is not None:
            output_files.append(f"ridgecrest_current_schema_{date_str}.csv")
        if consolidated is not None:
            output_files.append(f"consolidated_earthquake_observations_{date_str}.csv")
        output_files.append(report_file)
        
        print(f"Output files: {', '.join(output_files)}")
        total_records = 0
        if napa_current is not None:
            total_records += len(napa_current)
        if ridgecrest_current is not None:
            total_records += len(ridgecrest_current)
        print(f"Total records processed: {total_records}")
        
        # Show new columns created
        if ridgecrest_current is not None:
            new_columns = [col for col in ridgecrest_current.columns if col.startswith('_')]
            print(f"New columns created: {len(new_columns)}")
            print(f"New columns: {', '.join(sorted(new_columns))}")
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()